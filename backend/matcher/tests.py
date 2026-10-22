"""Tests for the resume matcher.

Nothing here calls the real Groq API. The real Groq SDK is used, but its HTTP
transport is replaced by FakeGroq, so request building, error classes and
response parsing are exercised for real. FakeGroq can answer differently on
every call, which is how the original bug (a different requirement list for
the same job description) is reproduced.

Run with:  python manage.py test matcher
"""

import json
import logging
import os
from unittest import mock

import groq
import httpx
import pymupdf
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, override_settings
from pydantic import ValidationError
from rest_framework.test import APIRequestFactory

from matcher import services
from matcher.groq_client import MODEL_NAME, get_groq_client
from matcher.schemas import CATEGORY_VALUES, JobRequirement
from matcher.services import TEST_JD
from matcher.views import analyze_resume_view, health_check


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

def req(text, category="skill", importance="high", kind="mandatory"):
    return {
        "requirement": text,
        "category": category,
        "importance": importance,
        "requirement_type": kind,
        "evidence": f"The job description says: {text}",
    }


# Includes responsibility and certification, the categories that used to make
# the pipeline crash.
REQUIREMENTS = [
    req("React.js"),
    req("Node.js"),
    req("HTML5 and CSS3"),
    req("RESTful API integration"),
    req("JavaScript ES6+", importance="medium"),
    req("Several years of web development experience", "experience"),
    req("Design, develop, test and maintain web applications",
        "responsibility", "medium"),
    req("Participate in code reviews", "responsibility", "low", "unspecified"),
    req("Bachelor's degree in computer science", "qualification", "low",
        "preferred"),
    req("AWS Certified Developer", "certification", "low", "preferred"),
]

# Two other answers the "model" might give for the same job description.
REQUIREMENTS_B = REQUIREMENTS[:6] + [
    req("TypeScript", importance="low", kind="preferred")
]
REQUIREMENTS_C = list(reversed(REQUIREMENTS[1:])) + [
    req("React.js", importance="medium")
]

RESUME_TEXT = (
    "Jane Doe - Senior Web Developer\n"
    "Seven years building React.js and Node.js applications with REST APIs.\n"
    "Skills: JavaScript ES6, HTML5, CSS3, TypeScript, AWS.\n"
    "B.Sc. Computer Science. Led code reviews for a team of six developers."
)


# ---------------------------------------------------------------------------
# A fake api.groq.com
# ---------------------------------------------------------------------------

def ok(content, finish_reason="stop"):
    if not isinstance(content, str):
        content = json.dumps(content)

    return httpx.Response(200, json={
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def error(status, code, message="error"):
    return httpx.Response(status, json={"error": {
        "message": message,
        "type": "invalid_request_error",
        "code": code,
    }})


def requested_ids(body):
    schema = body["response_format"]["json_schema"]["schema"]
    return schema["properties"]["evaluations"]["items"]["properties"]["id"]["enum"]


def evaluation_payload(body, status_for=lambda index: "demonstrated", **extra):
    payload = {
        "evaluations": [
            {
                "id": requirement_id,
                "status": status_for(index),
                "evidence": "The resume mentions it.",
            }
            for index, requirement_id in enumerate(requested_ids(body))
        ],
        "summary": "A solid match overall.",
        "strengths": ["Strong React background"],
        "improvements": [
            {"recommendation": "Add metrics", "reason": "Shows impact"}
        ],
        "keyword_suggestions": ["React.js"],
    }
    payload.update(extra)
    return payload


class FakeGroq:
    """Stands in for api.groq.com. Replace .extract / .evaluate per test.

    Each takes (call_number, request_body) and returns an httpx.Response.
    call_number counts from 0 separately for extraction and evaluation.
    """

    def __init__(self):
        self.requests = []
        self._counts = {"job_requirements": 0, "resume_analysis": 0}
        self.extract = lambda call, body: ok({"requirements": REQUIREMENTS})
        self.evaluate = lambda call, body: ok(evaluation_payload(body))

    def handler(self, request):
        body = json.loads(request.content)
        name = body["response_format"]["json_schema"]["name"]
        call = self._counts[name]
        self._counts[name] += 1
        self.requests.append(body)

        producer = self.extract if name == "job_requirements" else self.evaluate
        return producer(call, body)

    def calls(self, name):
        return [
            body for body in self.requests
            if body["response_format"]["json_schema"]["name"] == name
        ]

    @property
    def extractions(self):
        return self.calls("job_requirements")

    @property
    def evaluations(self):
        return self.calls("resume_analysis")

    def client(self):
        # max_retries=0: no sleeping in tests; the SDK's own retry of 429s and
        # 5xx is not what is being tested.
        return groq.Groq(
            api_key="test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
            max_retries=0,
        )


class MatcherTestCase(SimpleTestCase):
    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)

        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

        self.groq = FakeGroq()

        patchers = [
            mock.patch.object(
                services, "get_groq_client", side_effect=lambda: self.groq.client()
            ),
            # extract_job_requirements prints every raw model response.
            mock.patch("matcher.services.print", create=True),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def analyze(self, requirements=None, resume=RESUME_TEXT, jd=TEST_JD):
        return services.analyze_resume(
            resume_text=resume,
            job_description=jd,
            requirements=requirements or services.get_job_requirements(jd),
        )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SchemaTests(SimpleTestCase):
    def test_every_category_groq_may_return_is_accepted_by_pydantic(self):
        # Regression: schemas.py allowed 5 categories while the schema sent to
        # Groq allowed 10, so valid model output was rejected.
        groq_categories = (
            services.JD_REQUIREMENTS_SCHEMA["properties"]["requirements"]
            ["items"]["properties"]["category"]["enum"]
        )

        self.assertEqual(groq_categories, CATEGORY_VALUES)

        for category in groq_categories:
            JobRequirement(
                requirement="x", category=category, importance="high",
                requirement_type="mandatory", evidence="y",
            )

    def test_unknown_category_is_still_rejected(self):
        with self.assertRaises(ValidationError):
            JobRequirement(
                requirement="x", category="vibes", importance="high",
                requirement_type="mandatory", evidence="y",
            )

    def test_whitespace_only_text_is_rejected(self):
        with self.assertRaises(ValidationError):
            JobRequirement(
                requirement="   ", category="skill", importance="high",
                requirement_type="mandatory", evidence="y",
            )

    def test_evaluation_schema_only_allows_the_supplied_ids(self):
        schema = services._build_evaluation_schema(["R1", "R2"])
        item = schema["properties"]["evaluations"]["items"]["properties"]

        self.assertEqual(item["id"]["enum"], ["R1", "R2"])
        self.assertEqual(item["status"]["enum"],
                         ["demonstrated", "partial", "unclear"])
        # The model is no longer asked to echo category or importance.
        self.assertNotIn("category", item)
        self.assertNotIn("importance", item)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def scored(category, importance, status):
    return {"category": category, "importance": importance, "status": status}


class ScoreTests(SimpleTestCase):
    def test_original_weights_still_apply(self):
        # skill group 1.0 * 50, experience group 0.5 * 30 -> 65 / 80
        score = services.calculate_match_score([
            scored("skill", "high", "demonstrated"),
            scored("experience", "high", "partial"),
        ])
        self.assertEqual(score, 81)

    def test_importance_weights_requirements_inside_a_group(self):
        score = services.calculate_match_score([
            scored("skill", "high", "demonstrated"),
            scored("skill", "low", "unclear"),
        ])
        self.assertEqual(score, 75)

    def test_responsibility_counts_as_experience(self):
        # This category used to raise "Invalid requirement category".
        score = services.calculate_match_score([
            scored("responsibility", "high", "partial"),
            scored("skill", "high", "unclear"),
        ])
        self.assertEqual(score, 19)  # 0.5 * 30 / (30 + 50)

    def test_certification_counts_as_qualification(self):
        score = services.calculate_match_score([
            scored("certification", "high", "demonstrated"),
            scored("skill", "high", "unclear"),
        ])
        self.assertEqual(score, 17)  # 10 / (10 + 50)

    def test_location_counts_as_other(self):
        score = services.calculate_match_score([
            scored("location", "high", "demonstrated"),
            scored("skill", "high", "unclear"),
        ])
        self.assertEqual(score, 17)

    def test_every_category_can_be_scored(self):
        for category in CATEGORY_VALUES:
            score = services.calculate_match_score(
                [scored(category, "medium", "demonstrated")]
            )
            self.assertEqual(score, 100, category)

    def test_all_demonstrated_is_100_and_all_unclear_is_0(self):
        everything = [(c, "high") for c in CATEGORY_VALUES]
        self.assertEqual(services.calculate_match_score(
            [scored(c, i, "demonstrated") for c, i in everything]), 100)
        self.assertEqual(services.calculate_match_score(
            [scored(c, i, "unclear") for c, i in everything]), 0)

    def test_bad_values_are_rejected(self):
        for bad in (
            [],
            [scored("vibes", "high", "demonstrated")],
            [scored("skill", "urgent", "demonstrated")],
            [scored("skill", "high", "maybe")],
        ):
            with self.assertRaises(ValueError):
                services.calculate_match_score(bad)


# ---------------------------------------------------------------------------
# The reported bug: same JD, different requirements
# ---------------------------------------------------------------------------

class RequirementStabilityTests(MatcherTestCase):
    def varying_model(self):
        answers = [REQUIREMENTS, REQUIREMENTS_B, REQUIREMENTS_C]
        self.groq.extract = lambda call, body: ok(
            {"requirements": answers[call % len(answers)]}
        )

    def test_the_model_on_its_own_is_not_repeatable(self):
        # Control: shows the fake really does reproduce the reported problem.
        self.varying_model()

        first = services.normalize_requirements(
            services.extract_job_requirements(TEST_JD))
        second = services.normalize_requirements(
            services.extract_job_requirements(TEST_JD))

        self.assertNotEqual(first, second)

    def test_same_jd_gives_the_same_requirements_every_time(self):
        self.varying_model()

        results = [services.get_job_requirements(TEST_JD) for _ in range(5)]

        for later in results[1:]:
            self.assertEqual(later, results[0])
        self.assertEqual(len(self.groq.extractions), 1)

    def test_formatting_differences_reuse_the_same_extraction(self):
        self.varying_model()

        # Same words, different spacing: a non-breaking space instead of a
        # space, doubled spaces, Windows line endings, a zero-width space and
        # extra blank lines.
        reformatted = TEST_JD.replace("Strong experience", "Strong\u00a0experience")
        reformatted = (
            "\n\n" + reformatted.replace("\n", "\r\n").replace(" ", "  ")
            + "\u200b \n\n\n"
        )
        self.assertNotEqual(reformatted, TEST_JD)

        first = services.get_job_requirements(TEST_JD)
        second = services.get_job_requirements(reformatted)

        self.assertEqual(first, second)
        self.assertEqual(len(self.groq.extractions), 1)

    def test_a_different_jd_is_extracted_separately(self):
        services.get_job_requirements(TEST_JD)
        services.get_job_requirements(TEST_JD + "\nMust speak French.")

        self.assertEqual(len(self.groq.extractions), 2)

    def test_the_order_the_model_returns_does_not_matter(self):
        self.assertEqual(
            services.normalize_requirements(REQUIREMENTS),
            services.normalize_requirements(list(reversed(REQUIREMENTS))),
        )

    def test_too_few_requirements_is_an_error_and_is_not_remembered(self):
        answers = [REQUIREMENTS[:2], REQUIREMENTS]
        self.groq.extract = lambda call, body: ok({"requirements": answers[call]})

        with self.assertRaises(services.RequirementExtractionError):
            services.get_job_requirements(TEST_JD)

        # The bad answer was not cached: the next attempt asks again.
        self.assertEqual(len(services.get_job_requirements(TEST_JD)), 10)
        self.assertEqual(len(self.groq.extractions), 2)

    def test_too_many_requirements_is_an_error(self):
        many = [req(f"Skill number {n}") for n in range(41)]
        self.groq.extract = lambda call, body: ok({"requirements": many})

        with self.assertRaises(services.RequirementExtractionError):
            services.get_job_requirements(TEST_JD)

    def test_editing_the_prompt_stops_old_extractions_being_reused(self):
        services.get_job_requirements(TEST_JD)

        with mock.patch.object(
            services, "SYSTEM_PROMPT", services.SYSTEM_PROMPT + "\n20. New rule."
        ):
            services.get_job_requirements(TEST_JD)

        self.assertEqual(len(self.groq.extractions), 2)

    def test_empty_job_description_is_rejected(self):
        for bad in ("", "   \n", None):
            with self.assertRaises(ValueError):
                services.get_job_requirements(bad)


# ---------------------------------------------------------------------------
# Retries and failures when talking to the model
# ---------------------------------------------------------------------------

class RetryTests(MatcherTestCase):
    def extract_script(self, *replies):
        self.groq.extract = lambda call, body: replies[min(call, len(replies) - 1)]

    def test_invalid_json_is_retried(self):
        self.extract_script(ok("not json at all"),
                            ok({"requirements": REQUIREMENTS}))

        self.assertEqual(len(services.get_job_requirements(TEST_JD)), 10)
        self.assertEqual(len(self.groq.extractions), 2)

    def test_output_that_breaks_the_schema_is_retried(self):
        bad = [dict(REQUIREMENTS[0], category="vibes")]
        self.extract_script(ok({"requirements": bad}),
                            ok({"requirements": REQUIREMENTS}))

        self.assertEqual(len(services.get_job_requirements(TEST_JD)), 10)

    def test_groqs_json_validate_failed_is_retried(self):
        self.extract_script(
            error(400, "json_validate_failed",
                  "Generated JSON does not match the expected schema."),
            ok({"requirements": REQUIREMENTS}),
        )

        self.assertEqual(len(services.get_job_requirements(TEST_JD)), 10)
        self.assertEqual(len(self.groq.extractions), 2)

    def test_truncated_output_is_retried_with_more_room(self):
        self.extract_script(ok('{"requirements": [', finish_reason="length"),
                            ok({"requirements": REQUIREMENTS}))

        services.get_job_requirements(TEST_JD)

        first, second = self.groq.extractions
        self.assertGreater(
            second["max_completion_tokens"], first["max_completion_tokens"])

    def test_each_attempt_uses_a_different_seed(self):
        self.extract_script(ok("nope"), ok("nope"),
                            ok({"requirements": REQUIREMENTS}))

        services.get_job_requirements(TEST_JD)

        self.assertEqual([b["seed"] for b in self.groq.extractions], [42, 43, 44])

    def test_it_gives_up_after_three_attempts(self):
        self.extract_script(ok("nope"))

        with self.assertRaises(services.LLMOutputError):
            services.get_job_requirements(TEST_JD)

        self.assertEqual(len(self.groq.extractions), 3)

    def test_other_bad_requests_are_not_retried(self):
        self.extract_script(error(400, "invalid_request", "Bad parameter"))

        with self.assertRaises(groq.BadRequestError):
            services.get_job_requirements(TEST_JD)

        self.assertEqual(len(self.groq.extractions), 1)

    def test_authentication_errors_are_not_retried(self):
        self.extract_script(error(401, "invalid_api_key", "Invalid API Key"))

        with self.assertRaises(groq.AuthenticationError):
            services.get_job_requirements(TEST_JD)

        self.assertEqual(len(self.groq.extractions), 1)

    def test_rate_limits_are_not_retried_here(self):
        self.extract_script(error(429, "rate_limit_exceeded", "Slow down"))

        with self.assertRaises(groq.RateLimitError):
            services.get_job_requirements(TEST_JD)

        self.assertEqual(len(self.groq.extractions), 1)

    # Evaluation step

    def evaluate_script(self, *builders):
        self.groq.evaluate = lambda call, body: builders[min(call, len(builders) - 1)](body)

    def test_an_evaluation_missing_a_requirement_is_retried(self):
        def short(body):
            payload = evaluation_payload(body)
            payload["evaluations"].pop()
            return ok(payload)

        self.evaluate_script(short, lambda body: ok(evaluation_payload(body)))

        self.assertEqual(self.analyze()["overall_score"], 100)
        self.assertEqual(len(self.groq.evaluations), 2)

    def test_an_evaluation_that_repeats_a_requirement_is_retried(self):
        def repeated(body):
            payload = evaluation_payload(body)
            payload["evaluations"][-1] = dict(payload["evaluations"][0])
            return ok(payload)

        self.evaluate_script(repeated, lambda body: ok(evaluation_payload(body)))

        self.analyze()
        self.assertEqual(len(self.groq.evaluations), 2)

    def test_an_evaluation_naming_an_unknown_id_is_retried(self):
        def unknown(body):
            payload = evaluation_payload(body)
            payload["evaluations"][0]["id"] = "R99"
            return ok(payload)

        self.evaluate_script(unknown, lambda body: ok(evaluation_payload(body)))

        self.analyze()
        self.assertEqual(len(self.groq.evaluations), 2)

    def test_demonstrated_without_evidence_is_retried(self):
        def no_evidence(body):
            payload = evaluation_payload(body)
            payload["evaluations"][0]["evidence"] = ""
            return ok(payload)

        self.evaluate_script(no_evidence, lambda body: ok(evaluation_payload(body)))

        self.analyze()
        self.assertEqual(len(self.groq.evaluations), 2)

    def test_the_evaluation_gives_up_after_three_attempts(self):
        self.evaluate_script(lambda body: ok("nope"))

        with self.assertRaises(services.LLMOutputError):
            self.analyze()

        self.assertEqual(len(self.groq.evaluations), 3)


# ---------------------------------------------------------------------------
# Resume analysis
# ---------------------------------------------------------------------------

class AnalysisTests(MatcherTestCase):
    def test_result_has_the_shape_the_frontend_reads(self):
        result = self.analyze()

        self.assertEqual(
            set(result),
            {"overall_score", "summary", "requirements", "matched_skills",
             "missing_skills", "strengths", "improvements",
             "keyword_suggestions"},
        )
        self.assertIsInstance(result["overall_score"], int)
        self.assertTrue(0 <= result["overall_score"] <= 100)
        self.assertIsInstance(result["summary"], str)

        for item in result["requirements"]:
            self.assertEqual(
                set(item),
                {"requirement", "category", "importance",
                 "requirement_type", "evidence", "status"},
            )
        for item in result["matched_skills"]:
            self.assertEqual(set(item), {"skill", "evidence"})
        for item in result["missing_skills"]:
            self.assertEqual(set(item), {"skill", "importance"})
        for item in result["improvements"]:
            self.assertEqual(set(item), {"recommendation", "reason"})

    def test_a_job_with_responsibilities_and_certifications_works(self):
        # Regression: this raised "Invalid requirement category" before.
        result = self.analyze()

        categories = {r["category"] for r in result["requirements"]}
        self.assertIn("responsibility", categories)
        self.assertIn("certification", categories)
        self.assertEqual(result["overall_score"], 100)

    def test_requirements_come_from_python_not_the_model(self):
        expected = services.normalize_requirements(REQUIREMENTS)

        result = self.analyze()

        for given, returned in zip(expected, result["requirements"]):
            for field in ("requirement", "category", "importance",
                          "requirement_type"):
                self.assertEqual(returned[field], given[field])
        self.assertEqual(len(result["requirements"]), len(expected))

    def test_the_order_requirements_are_passed_in_does_not_matter(self):
        first = self.analyze(requirements=REQUIREMENTS)
        second = self.analyze(requirements=list(reversed(REQUIREMENTS)))

        self.assertEqual(first, second)
        self.assertEqual(len(self.groq.evaluations), 1)

    def test_the_callers_requirements_are_not_modified(self):
        before = json.dumps(REQUIREMENTS)
        self.analyze(requirements=REQUIREMENTS)
        self.assertEqual(json.dumps(REQUIREMENTS), before)

    def varying_evaluator(self):
        # First answer: everything demonstrated. Every later answer: nothing.
        self.groq.evaluate = lambda call, body: ok(evaluation_payload(
            body, status_for=lambda i: "demonstrated" if call == 0 else "unclear"
        ))

    def test_same_resume_and_jd_gives_the_same_score_every_time(self):
        self.varying_evaluator()

        scores = [self.analyze()["overall_score"] for _ in range(5)]

        self.assertEqual(scores, [100] * 5)
        self.assertEqual(len(self.groq.evaluations), 1)

    def test_a_different_resume_is_evaluated_again(self):
        self.varying_evaluator()

        self.analyze()
        other = self.analyze(resume=RESUME_TEXT + "\nAlso knows Rust.")

        self.assertEqual(len(self.groq.evaluations), 2)
        self.assertEqual(other["overall_score"], 0)

    @override_settings(MATCHER_ANALYSIS_CACHE_TTL=0)
    def test_remembering_evaluations_can_be_switched_off(self):
        self.varying_evaluator()

        first = self.analyze()["overall_score"]
        second = self.analyze()["overall_score"]

        self.assertEqual((first, second), (100, 0))
        self.assertEqual(len(self.groq.evaluations), 2)

    def test_changing_scoring_code_is_not_hidden_by_the_cache(self):
        # Only the model's answer is remembered; the score is recomputed.
        # Skills demonstrated, everything else unclear.
        canonical = services.normalize_requirements(REQUIREMENTS)
        self.groq.evaluate = lambda call, body: ok(evaluation_payload(
            body,
            status_for=lambda i: (
                "demonstrated" if canonical[i]["category"] == "skill" else "unclear"
            ),
        ))

        before = self.analyze()["overall_score"]

        # Count skills in the weight-10 "other" group instead of weight 50.
        with mock.patch.dict(services.SCORING_GROUP, {"skill": "other"}):
            after = self.analyze()["overall_score"]

        self.assertEqual(before, 56)  # 50 / (50 + 30 + 10)
        self.assertEqual(after, 20)   # 10 / (10 + 30 + 10)
        self.assertEqual(len(self.groq.evaluations), 1)

    def test_each_evaluation_is_matched_to_the_right_requirement(self):
        canonical = services.normalize_requirements(REQUIREMENTS)

        def tagged(body):
            payload = evaluation_payload(body)
            # Answer in reverse order, with evidence that names its own id.
            payload["evaluations"] = [
                {"id": rid, "status": "partial", "evidence": f"evidence for {rid}"}
                for rid in reversed(requested_ids(body))
            ]
            return ok(payload)

        self.groq.evaluate = lambda call, body: tagged(body)

        result = self.analyze()

        for number, item in enumerate(result["requirements"], start=1):
            self.assertEqual(item["requirement"], canonical[number - 1]["requirement"])
            self.assertEqual(item["evidence"], f"evidence for R{number}")
            self.assertEqual(item["status"], "partial")

    def test_extra_list_items_are_trimmed_not_rejected(self):
        self.groq.evaluate = lambda call, body: ok(evaluation_payload(
            body,
            strengths=[f"strength {n}" for n in range(7)],
            improvements=[{"recommendation": f"r{n}", "reason": "x"}
                          for n in range(6)],
            keyword_suggestions=[f"kw{n}" for n in range(15)],
        ))

        result = self.analyze()

        self.assertEqual(len(result["strengths"]), 4)
        self.assertEqual(len(result["improvements"]), 4)
        self.assertEqual(len(result["keyword_suggestions"]), 10)

    def test_unclear_without_evidence_gets_default_text(self):
        def unclear(body):
            payload = evaluation_payload(body, status_for=lambda i: "unclear")
            for item in payload["evaluations"]:
                item["evidence"] = ""
            return ok(payload)

        self.groq.evaluate = lambda call, body: unclear(body)

        result = self.analyze()

        self.assertEqual(result["overall_score"], 0)
        for item in result["requirements"]:
            self.assertEqual(item["evidence"], services.NO_EVIDENCE_TEXT)

    def test_skill_lists_agree_with_the_requirement_statuses(self):
        # Every third requirement is demonstrated, every third partial,
        # every third unclear.
        statuses = ["demonstrated", "partial", "unclear"]
        self.groq.evaluate = lambda call, body: ok(evaluation_payload(
            body, status_for=lambda i: statuses[i % 3]
        ))

        result = self.analyze()

        by_name = {r["requirement"]: r for r in result["requirements"]}
        self.assertTrue(result["matched_skills"])
        self.assertTrue(result["missing_skills"])

        for item in result["matched_skills"]:
            self.assertEqual(by_name[item["skill"]]["status"], "demonstrated")
            self.assertEqual(item["evidence"], by_name[item["skill"]]["evidence"])
        for item in result["missing_skills"]:
            self.assertNotEqual(by_name[item["skill"]]["status"], "demonstrated")
            self.assertEqual(item["importance"], by_name[item["skill"]]["importance"])

        self.assertLessEqual(len(result["matched_skills"]), 5)
        self.assertLessEqual(len(result["missing_skills"]), 5)

    def test_skills_are_listed_before_responsibilities(self):
        self.groq.evaluate = lambda call, body: ok(evaluation_payload(
            body, status_for=lambda i: "unclear"
        ))

        result = self.analyze()

        missing = [item["skill"] for item in result["missing_skills"]]
        categories = {r["requirement"]: r["category"]
                      for r in result["requirements"]}

        self.assertEqual({categories[name] for name in missing}, {"skill"})

    def test_requests_sent_to_groq(self):
        self.analyze()

        extraction = self.groq.extractions[0]
        self.assertEqual(extraction["model"], MODEL_NAME)
        self.assertEqual(extraction["temperature"], 0)
        self.assertEqual(extraction["seed"], 42)
        self.assertEqual(extraction["reasoning_effort"], "low")
        self.assertEqual(extraction["max_completion_tokens"], 5000)
        self.assertEqual(extraction["messages"][0],
                         {"role": "system", "content": services.SYSTEM_PROMPT})
        self.assertIn("<job_description>", extraction["messages"][1]["content"])
        self.assertIn("React.js: with several years",
                      extraction["messages"][1]["content"])

        rf = extraction["response_format"]
        self.assertEqual(rf["type"], "json_schema")
        self.assertIs(rf["json_schema"]["strict"], True)
        self.assertEqual(rf["json_schema"]["schema"],
                         services.JD_REQUIREMENTS_SCHEMA)

        evaluation = self.groq.evaluations[0]
        self.assertEqual(evaluation["model"], MODEL_NAME)
        self.assertEqual(evaluation["temperature"], 0)
        self.assertEqual(evaluation["seed"], 42)
        # The original evaluation call did not set reasoning_effort.
        self.assertNotIn("reasoning_effort", evaluation)
        self.assertEqual(evaluation["max_completion_tokens"], 5000)
        self.assertEqual(requested_ids(evaluation),
                         [f"R{n}" for n in range(1, 11)])

        prompt = evaluation["messages"][1]["content"]
        self.assertIn("<resume>\n" + RESUME_TEXT + "\n</resume>", prompt)
        self.assertIn("<requirements>", prompt)
        self.assertIn('"id": "R1"', prompt)
        self.assertIn("exactly 10 requirements", prompt)

    def test_bad_input_is_rejected(self):
        for resume, jd, reqs in (
            ("", TEST_JD, REQUIREMENTS),
            ("   ", TEST_JD, REQUIREMENTS),
            (RESUME_TEXT, "", REQUIREMENTS),
            (RESUME_TEXT, TEST_JD, []),
        ):
            with self.assertRaises(ValueError):
                services.analyze_resume(
                    resume_text=resume, job_description=jd, requirements=reqs)


# ---------------------------------------------------------------------------
# The API endpoint
# ---------------------------------------------------------------------------

def make_pdf(text=RESUME_TEXT, pages=1, password=None):
    document = pymupdf.open()
    for _ in range(pages):
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=10)

    options = {}
    if password:
        options = {
            "encryption": pymupdf.PDF_ENCRYPT_AES_256,
            "user_pw": password,
            "owner_pw": password + "-owner",
        }

    data = document.tobytes(**options)
    document.close()
    return data


factory = APIRequestFactory()


class ViewTests(MatcherTestCase):
    def post(self, pdf=None, jd=TEST_JD, name="cv.pdf"):
        data = {}
        if pdf is not None:
            data["resume"] = SimpleUploadedFile(
                name, pdf, content_type="application/pdf")
        if jd is not None:
            data["job_description"] = jd

        return analyze_resume_view(
            factory.post("/analyze/", data, format="multipart"))

    def test_health_check(self):
        response = health_check(factory.get("/health/"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["status"], "success")

    def test_the_same_pdf_and_jd_five_times_gives_one_score(self):
        # The reported problem: the model answers differently every time.
        answers = [REQUIREMENTS, REQUIREMENTS_B, REQUIREMENTS_C]
        self.groq.extract = lambda call, body: ok(
            {"requirements": answers[call % 3]})
        statuses = ["demonstrated", "partial", "unclear"]
        self.groq.evaluate = lambda call, body: ok(evaluation_payload(
            body, status_for=lambda i: statuses[(i + call) % 3]))

        pdf = make_pdf()
        responses = [self.post(pdf) for _ in range(5)]

        self.assertEqual({r.status_code for r in responses}, {200})
        first = responses[0].data
        self.assertEqual(first["status"], "success")
        self.assertEqual(first["filename"], "cv.pdf")

        for response in responses[1:]:
            self.assertEqual(response.data, first)

        # One extraction and one evaluation served all five requests.
        self.assertEqual(len(self.groq.extractions), 1)
        self.assertEqual(len(self.groq.evaluations), 1)

        # Control: with nothing remembered the same input does come back
        # different, so the check above really is testing the fix.
        cache.clear()
        again = self.post(pdf).data
        self.assertNotEqual(
            again["analysis"]["requirements"], first["analysis"]["requirements"])

    def test_the_original_input_checks_still_apply(self):
        pdf = make_pdf()
        cases = [
            ("no resume", self.post(None), "Please upload a resume PDF."),
            ("not a pdf", self.post(pdf, name="cv.docx"),
             "Only PDF files are supported."),
            ("too big", self.post(b"0" * (5 * 1024 * 1024 + 1)),
             "The PDF must be 5 MB or smaller."),
            ("no jd", self.post(pdf, jd=None),
             "Please provide a job description."),
            ("blank jd", self.post(pdf, jd="   "),
             "Please provide a job description."),
            ("long jd", self.post(pdf, jd="x" * 20001),
             "The job description is too long."),
            ("password", self.post(make_pdf(password="secret")),
             "Password-protected PDFs are not supported."),
            ("21 pages", self.post(make_pdf(pages=21)),
             "The PDF must contain 1–20 pages."),
            ("no text", self.post(make_pdf(text=" ")),
             "Insufficient readable text was found. "
             "The PDF may be scanned or image-based."),
            ("garbage", self.post(b"this is not a pdf"),
             "Unable to read this PDF. Please check the file."),
        ]

        for label, response, message in cases:
            self.assertEqual(response.status_code, 400, label)
            self.assertEqual(response.data, {"error": message}, label)

        self.assertEqual(self.groq.requests, [])

    def test_a_job_description_with_too_few_requirements_is_a_400(self):
        self.groq.extract = lambda call, body: ok(
            {"requirements": REQUIREMENTS[:2]})

        response = self.post(make_pdf(), jd="Python developer")

        self.assertEqual(response.status_code, 400)
        self.assertIn("Not enough requirements", response.data["error"])

    def test_a_rate_limit_is_a_429(self):
        self.groq.extract = lambda call, body: error(429, "rate_limit_exceeded")

        response = self.post(make_pdf())

        self.assertEqual(response.status_code, 429)
        self.assertIn("try again", response.data["error"])

    def test_unusable_model_output_is_the_original_502(self):
        self.groq.extract = lambda call, body: ok("nope")

        response = self.post(make_pdf())

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.data, {
            "error": "Resume analysis failed. Please try again "
                     "or check the backend configuration."
        })

    def test_a_missing_api_key_is_a_502_that_does_not_leak_details(self):
        with mock.patch.object(
            services, "get_groq_client",
            side_effect=RuntimeError("GROQ_API_KEY is missing."),
        ):
            response = self.post(make_pdf())

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("GROQ_API_KEY", response.data["error"])

    def test_an_invalid_api_key_is_a_502(self):
        self.groq.extract = lambda call, body: error(401, "invalid_api_key")

        self.assertEqual(self.post(make_pdf()).status_code, 502)

    def test_a_failure_in_the_evaluation_step_is_a_502(self):
        self.groq.evaluate = lambda call, body: ok("nope")

        self.assertEqual(self.post(make_pdf()).status_code, 502)

    def test_failures_are_not_remembered(self):
        replies = [ok("nope"), ok("nope"), ok("nope")]
        self.groq.extract = lambda call, body: (
            replies[call] if call < 3 else ok({"requirements": REQUIREMENTS}))

        pdf = make_pdf()
        self.assertEqual(self.post(pdf).status_code, 502)
        self.assertEqual(self.post(pdf).status_code, 200)


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

class GroqClientTests(SimpleTestCase):
    def test_groq_api_key_is_used(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "from-groq"}, clear=True):
            self.assertEqual(get_groq_client().api_key, "from-groq")

    def test_api_key_is_still_accepted(self):
        with mock.patch.dict(os.environ, {"API_KEY": "legacy"}, clear=True):
            self.assertEqual(get_groq_client().api_key, "legacy")

    def test_groq_api_key_wins_when_both_are_set(self):
        env = {"GROQ_API_KEY": "new", "API_KEY": "old"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_groq_client().api_key, "new")

    def test_a_missing_key_raises_a_clear_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY is missing"):
                get_groq_client()
