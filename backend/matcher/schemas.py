from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field


# These Literals are the single source of truth for the allowed values.
# services.py builds the enums of the JSON schemas it sends to Groq from the
# *_VALUES lists below, so what the model may return and what Pydantic accepts
# can no longer drift apart.
#
# (Before, this file allowed 5 categories while the schema sent to Groq allowed
# 10 and the evaluation schema only 4, so valid model output was rejected.)
RequirementCategory = Literal[
    "skill",
    "experience",
    "qualification",
    "certification",
    "responsibility",
    "domain_knowledge",
    "language",
    "work_authorization",
    "location",
    "other",
]

RequirementImportance = Literal[
    "high",
    "medium",
    "low",
]

RequirementType = Literal[
    "mandatory",
    "preferred",
    "unspecified",
]

EvaluationStatus = Literal[
    "demonstrated",
    "partial",
    "unclear",
]

CATEGORY_VALUES = list(get_args(RequirementCategory))
IMPORTANCE_VALUES = list(get_args(RequirementImportance))
REQUIREMENT_TYPE_VALUES = list(get_args(RequirementType))
STATUS_VALUES = list(get_args(EvaluationStatus))


class JobRequirement(BaseModel):
    # Whitespace-only text is treated as empty (and therefore rejected).
    model_config = ConfigDict(str_strip_whitespace=True)

    requirement: str = Field(min_length=1)
    category: RequirementCategory
    importance: RequirementImportance
    requirement_type: RequirementType
    evidence: str = Field(min_length=1)


class JobRequirements(BaseModel):
    requirements: list[JobRequirement]


# What the model returns when it evaluates a resume. It refers to requirements
# by id, so it never has to copy requirement text back exactly.
class RequirementEvaluation(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(min_length=1)
    status: EvaluationStatus
    # May be empty only when status == "unclear"; services.py enforces that.
    evidence: str


class ImprovementSuggestion(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    recommendation: str = Field(min_length=1)
    reason: str


class ResumeEvaluation(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    evaluations: list[RequirementEvaluation]
    summary: str = Field(min_length=1)
    strengths: list[str]
    improvements: list[ImprovementSuggestion]
    keyword_suggestions: list[str]
