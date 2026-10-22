"use client";

import { useRef, useState } from "react";
import type { FormEvent } from "react";
import Link from "next/link";
import Image from "next/image";

type Requirement = {
    requirement: string;
    category: "skill" | "experience" | "qualification" | "other";
    importance: "high" | "medium" | "low";
    status: "demonstrated" | "partial" | "unclear";
    evidence: string;
};

type MatchedSkill = {
    skill: string;
    evidence: string;
};

type MissingSkill = {
    skill: string;
    importance: "high" | "medium" | "low";
};

type Improvement = {
    recommendation: string;
    reason: string;
};

type AnalysisResult = {
    overall_score: number;
    summary: string;
    requirements: Requirement[];
    matched_skills: MatchedSkill[];
    missing_skills: MissingSkill[];
    strengths: string[];
    improvements: Improvement[];
    keyword_suggestions: string[];
};

const API_BASE_URL =
    process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export default function Home() {
    const fileInputRef = useRef<HTMLInputElement>(null);

    const [file, setFile] = useState<File | null>(null);
    const [jobDescription, setJobDescription] = useState("");
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);
    const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);

    function handleFileChange(selectedFile?: File) {
        setError("");
        setAnalysis(null);

        if (!selectedFile) return;

        if (
            selectedFile.type !== "application/pdf" &&
            !selectedFile.name.toLowerCase().endsWith(".pdf")
        ) {
            setError("Please select a PDF file.");
            return;
        }

        if (selectedFile.size > 5 * 1024 * 1024) {
            setError("Your PDF must be 5 MB or smaller.");
            return;
        }

        setFile(selectedFile);
    }

    async function handleSubmit(event: FormEvent<HTMLFormElement>) {
        event.preventDefault();
        setError("");
        setAnalysis(null);

        if (!file) {
            setError("Please upload your resume PDF.");
            return;
        }

        if (!jobDescription.trim()) {
            setError("Please enter the job description.");
            return;
        }

        setLoading(true);

        try {
            const formData = new FormData();
            formData.append("resume", file);
            formData.append("job_description", jobDescription.trim());

            const response = await fetch(`${API_BASE_URL}/api/analyze/`, {
                method: "POST",
                body: formData,
            });

            const payload = await response.json().catch(() => null);

            if (!response.ok) {
                throw new Error(
                    typeof payload?.error === "string"
                        ? payload?.error
                        : `Analysis failed (HTTP ${response?.status}). Please try again.`,
                );
            }

            if (!payload?.analysis) {
                throw new Error("The server returned an unexpected response.");
            }

            setAnalysis(payload.analysis as AnalysisResult);
        } catch (err) {
            setError(
                err instanceof TypeError
                    ? "Cannot connect to Django. Check that the backend is running and CORS is configured."
                    : err instanceof Error
                      ? err.message
                      : "Something went wrong. Please try again.",
            );
        } finally {
            setLoading(false);
        }
    }

    return (
        <main className="min-h-screen bg-slate-50 text-slate-900">
            <header className="border-b border-slate-200 bg-white">
                <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-5">
                    <Link href="/" className="flex items-center gap-3">
                        <Image
                            src="/images/logo.png"
                            alt="Company Logo"
                            className="rounded"
                            width={25}
                            height={20}
                        />
                        <span className="text-lg font-bold tracking-tight">
                            Resume<span className="text-blue-600">Match</span>
                        </span>
                    </Link>
                    <span className="rounded-full truncate max-[480px]:w-1/3 border border-blue-100 bg-blue-50 px-3 py-1.5 text-xs font-semibold text-blue-700">
                        AI-powered analysis
                    </span>
                </div>
            </header>

            <section className="mx-auto max-w-4xl px-5 pb-16 pt-12 sm:pt-16">
                <div className="mb-10 text-center">
                    <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-blue-100 bg-white px-4 py-2 text-sm text-blue-700 shadow-sm">
                        ✦ Make every application count
                    </div>
                    <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
                        Your next role starts with
                        <span className="mt-2 block text-blue-600">
                            a stronger resume.
                        </span>
                    </h1>
                    <p className="mx-auto mt-5 max-w-2xl text-base leading-7 text-slate-600 sm:text-lg">
                        Compare your resume with a job description to discover
                        matching skills, identify gaps, and get practical
                        recommendations.
                    </p>
                </div>

                <form
                    onSubmit={handleSubmit}
                    className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xl shadow-slate-200/50 sm:p-8"
                >
                    <div className="mb-7">
                        <div className="mb-3 flex items-center gap-3">
                            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-50 font-semibold text-blue-700">
                                1
                            </span>
                            <div>
                                <h2 className="font-semibold">
                                    Upload your resume
                                </h2>
                                <p className="text-sm text-slate-500">
                                    PDF format · Maximum 5 MB
                                </p>
                            </div>
                        </div>

                        <button
                            type="button"
                            onClick={() => fileInputRef.current?.click()}
                            onDragOver={(event) => event.preventDefault()}
                            onDrop={(event) => {
                                event.preventDefault();
                                handleFileChange(event.dataTransfer.files[0]);
                            }}
                            className="flex w-full flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 px-4 py-9 text-center transition hover:border-blue-400 hover:bg-blue-50/50"
                        >
                            <span className="mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-100 text-2xl">
                                ↑
                            </span>
                            <span className="font-semibold text-slate-800">
                                {file
                                    ? file?.name
                                    : "Click to upload or drag your PDF here"}
                            </span>
                            <span className="mt-2 text-sm text-slate-500">
                                {file
                                    ? `${(file?.size / 1024).toFixed(0)} KB · PDF selected`
                                    : "Upload your resume to get started"}
                            </span>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept=".pdf,application/pdf"
                                className="hidden"
                                onChange={(event) =>
                                    handleFileChange(event.target.files?.[0])
                                }
                            />
                        </button>

                        {file && (
                            <button
                                type="button"
                                onClick={() => {
                                    setFile(null);
                                    setAnalysis(null);
                                    if (fileInputRef.current) {
                                        fileInputRef.current.value = "";
                                    }
                                }}
                                className="mt-3 text-sm font-medium text-red-600 hover:text-red-700"
                            >
                                Remove file
                            </button>
                        )}
                    </div>

                    <div className="mb-7">
                        <div className="mb-3 flex items-center gap-3">
                            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-50 font-semibold text-blue-700">
                                2
                            </span>
                            <div>
                                <h2 className="font-semibold">
                                    Add the job description
                                </h2>
                                <p className="text-sm text-slate-500">
                                    Include the responsibilities and required
                                    skills.
                                </p>
                            </div>
                        </div>

                        <textarea
                            value={jobDescription}
                            onChange={(event) => {
                                setJobDescription(event.target.value);
                                setAnalysis(null);
                            }}
                            placeholder="Paste the job description here..."
                            rows={8}
                            maxLength={20000}
                            className="w-full resize-y rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm leading-6 outline-none transition placeholder:text-slate-400 focus:border-blue-500 focus:ring-4 focus:ring-blue-100"
                        />
                        <p className="mt-2 text-right text-xs text-slate-500">
                            {jobDescription?.length.toLocaleString()} / 20,000
                            characters
                        </p>
                    </div>

                    {error && (
                        <p
                            role="alert"
                            className="mb-5 rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-800"
                        >
                            {error}
                        </p>
                    )}

                    <button
                        type="submit"
                        disabled={loading}
                        className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-4 font-semibold text-white shadow-lg shadow-blue-600/20 transition hover:bg-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-200 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                        {loading ? (
                            <>
                                <span className="animate-spin">◌</span>
                                Analyzing your resume...
                            </>
                        ) : (
                            <>
                                Analyze my resume{" "}
                                <span aria-hidden="true">→</span>
                            </>
                        )}
                    </button>

                    <p className="mt-4 text-center text-xs leading-5 text-slate-500">
                        AI-generated feedback can be imperfect. Verify
                        suggestions against your actual experience.
                    </p>
                </form>

                {analysis && (
                    <section className="mt-10 space-y-6">
                        {/*
                        Group demonstrated requirements separately from requirements
                        that need clearer or additional evidence.
                        */}
                        {(() => {
                            const statusStyles: Record<
                                Requirement["status"],
                                string
                            > = {
                                demonstrated:
                                    "bg-emerald-50 text-emerald-700 ring-emerald-200",
                                partial:
                                    "bg-amber-50 text-amber-700 ring-amber-200",
                                unclear:
                                    "bg-slate-100 text-slate-700 ring-slate-200",
                            };

                            const importanceStyles: Record<
                                Requirement["importance"],
                                string
                            > = {
                                high: "bg-red-50 text-red-700",
                                medium: "bg-amber-50 text-amber-700",
                                low: "bg-slate-100 text-slate-600",
                            };

                            const importanceOrder = {
                                high: 0,
                                medium: 1,
                                low: 2,
                            };

                            const strongRequirements =
                                analysis.requirements.filter(
                                    (item) => item.status === "demonstrated",
                                );

                            const improvementRequirements =
                                analysis.requirements
                                    .filter(
                                        (item) =>
                                            item.status !== "demonstrated",
                                    )
                                    .sort(
                                        (a, b) =>
                                            importanceOrder[a.importance] -
                                            importanceOrder[b.importance],
                                    );

                            return (
                                <>
                                    {/* Overall match */}
                                    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
                                        <div className="p-6 sm:p-8">
                                            <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
                                                <div className="flex-1">
                                                    <p className="text-sm font-semibold uppercase tracking-wider text-blue-600">
                                                        Resume analysis
                                                    </p>

                                                    <h2 className="mt-2 text-2xl font-bold tracking-tight text-slate-900">
                                                        Your match overview
                                                    </h2>

                                                    <p className="mt-3 max-w-xl leading-7 text-slate-600">
                                                        {analysis.summary}
                                                    </p>
                                                </div>

                                                <div className="flex h-28 w-28 shrink-0 flex-col items-center justify-center self-center rounded-full border-8 border-blue-50 bg-white sm:self-auto">
                                                    <span className="text-3xl font-bold text-blue-600">
                                                        {analysis.overall_score}
                                                        %
                                                    </span>
                                                    <span className="text-xs font-medium text-slate-500">
                                                        Match score
                                                    </span>
                                                </div>
                                            </div>

                                            <div className="mt-6">
                                                <div className="mb-2 flex justify-between text-sm">
                                                    <span className="font-medium text-slate-700">
                                                        Overall compatibility
                                                    </span>
                                                    <span className="text-slate-500">
                                                        {analysis.overall_score}{" "}
                                                        / 100
                                                    </span>
                                                </div>

                                                <div
                                                    className="h-2.5 overflow-hidden rounded-full bg-slate-100"
                                                    role="progressbar"
                                                    aria-label="Overall match score"
                                                    aria-valuemin={0}
                                                    aria-valuemax={100}
                                                    aria-valuenow={
                                                        analysis.overall_score
                                                    }
                                                >
                                                    <div
                                                        className="h-full rounded-full bg-blue-600 transition-all duration-700"
                                                        style={{
                                                            width: `${Math.max(
                                                                0,
                                                                Math.min(
                                                                    100,
                                                                    analysis.overall_score,
                                                                ),
                                                            )}%`,
                                                        }}
                                                    />
                                                </div>
                                            </div>

                                            <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
                                                <div className="rounded-xl bg-emerald-50 p-4">
                                                    <p className="text-2xl font-bold text-emerald-700">
                                                        {
                                                            strongRequirements.length
                                                        }
                                                    </p>
                                                    <p className="mt-1 text-sm text-emerald-800">
                                                        Demonstrated
                                                    </p>
                                                </div>

                                                <div className="rounded-xl bg-amber-50 p-4">
                                                    <p className="text-2xl font-bold text-amber-700">
                                                        {
                                                            analysis.requirements.filter(
                                                                (item) =>
                                                                    item.status ===
                                                                    "partial",
                                                            ).length
                                                        }
                                                    </p>
                                                    <p className="mt-1 text-sm text-amber-800">
                                                        Partially demonstrated
                                                    </p>
                                                </div>

                                                <div className="rounded-xl bg-slate-100 p-4">
                                                    <p className="text-2xl font-bold text-slate-700">
                                                        {
                                                            analysis.requirements.filter(
                                                                (item) =>
                                                                    item.status ===
                                                                    "unclear",
                                                            ).length
                                                        }
                                                    </p>
                                                    <p className="mt-1 text-sm text-slate-600">
                                                        Evidence unclear
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Already Strong */}
                                    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
                                        <div className="mb-6 flex items-start gap-3">
                                            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-100 text-xl text-emerald-700">
                                                ✓
                                            </span>

                                            <div>
                                                <h2 className="text-xl font-bold text-slate-900">
                                                    Already Strong
                                                </h2>
                                                <p className="mt-1 text-sm leading-6 text-slate-500">
                                                    Requirements supported by
                                                    evidence in your resume.
                                                </p>
                                            </div>
                                        </div>

                                        {strongRequirements.length > 0 ? (
                                            <div className="space-y-3">
                                                {strongRequirements.map(
                                                    (item, index) => (
                                                        <article
                                                            key={`${item.requirement}-${index}`}
                                                            className="rounded-xl border border-emerald-100 bg-emerald-50/40 p-4"
                                                        >
                                                            <div className="flex flex-wrap items-center justify-between gap-3">
                                                                <h3 className="font-semibold text-slate-900">
                                                                    {
                                                                        item.requirement
                                                                    }
                                                                </h3>

                                                                <span
                                                                    className={`rounded-full px-3 py-1 text-xs font-medium capitalize ring-1 ring-inset ${statusStyles[item.status]}`}
                                                                >
                                                                    {
                                                                        item.status
                                                                    }
                                                                </span>
                                                            </div>

                                                            <p className="mt-2 text-sm leading-6 text-slate-600">
                                                                <span className="font-medium text-slate-800">
                                                                    Resume
                                                                    evidence:{" "}
                                                                </span>
                                                                {item.evidence}
                                                            </p>

                                                            <div className="mt-3 flex flex-wrap gap-2">
                                                                <span className="rounded-md bg-white px-2.5 py-1 text-xs capitalize text-slate-600 ring-1 ring-slate-200">
                                                                    {
                                                                        item.category
                                                                    }
                                                                </span>

                                                                <span
                                                                    className={`rounded-md px-2.5 py-1 text-xs capitalize ${importanceStyles[item.importance]}`}
                                                                >
                                                                    {
                                                                        item.importance
                                                                    }{" "}
                                                                    importance
                                                                </span>
                                                            </div>
                                                        </article>
                                                    ),
                                                )}
                                            </div>
                                        ) : (
                                            <p className="rounded-xl bg-slate-50 p-4 text-sm leading-6 text-slate-600">
                                                No requirements were
                                                conclusively demonstrated in the
                                                supplied resume. Review the
                                                evidence and make sure your
                                                relevant experience is included.
                                            </p>
                                        )}
                                    </div>

                                    {/* Required Improvements */}
                                    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
                                        <div className="mb-6 flex items-start gap-3">
                                            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-amber-100 text-xl text-amber-700">
                                                ↗
                                            </span>

                                            <div>
                                                <h2 className="text-xl font-bold text-slate-900">
                                                    Required Improvements
                                                </h2>
                                                <p className="mt-1 text-sm leading-6 text-slate-500">
                                                    Requirements needing
                                                    stronger evidence,
                                                    prioritized by importance.
                                                </p>
                                            </div>
                                        </div>

                                        {improvementRequirements.length > 0 ? (
                                            <div className="space-y-3">
                                                {improvementRequirements.map(
                                                    (item, index) => (
                                                        <article
                                                            key={`${item.requirement}-${index}`}
                                                            className="rounded-xl border border-slate-200 p-4 transition hover:border-blue-200"
                                                        >
                                                            <div className="flex flex-wrap items-start justify-between gap-3">
                                                                <div>
                                                                    <h3 className="font-semibold text-slate-900">
                                                                        {
                                                                            item.requirement
                                                                        }
                                                                    </h3>

                                                                    <p className="mt-1 text-sm capitalize text-slate-500">
                                                                        {
                                                                            item.category
                                                                        }
                                                                    </p>
                                                                </div>

                                                                <span
                                                                    className={`rounded-full px-3 py-1 text-xs font-medium capitalize ring-1 ring-inset ${statusStyles[item.status]}`}
                                                                >
                                                                    {item.status ===
                                                                    "unclear"
                                                                        ? "Evidence unclear"
                                                                        : "Partial evidence"}
                                                                </span>
                                                            </div>

                                                            <p className="mt-3 text-sm leading-6 text-slate-600">
                                                                {item.status ===
                                                                "unclear"
                                                                    ? "Your resume does not provide enough evidence to confirm this requirement. If you have relevant experience, consider adding a specific project or achievement."
                                                                    : "Your resume contains some related evidence, but it may not fully demonstrate this requirement. Add specific examples where applicable."}
                                                            </p>

                                                            {item.evidence && (
                                                                <p className="mt-3 rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-600">
                                                                    <span className="font-medium text-slate-800">
                                                                        Existing
                                                                        evidence:{" "}
                                                                    </span>
                                                                    {
                                                                        item.evidence
                                                                    }
                                                                </p>
                                                            )}

                                                            <div className="mt-3 flex flex-wrap gap-2">
                                                                <span
                                                                    className={`rounded-md px-2.5 py-1 text-xs font-medium capitalize ${importanceStyles[item.importance]}`}
                                                                >
                                                                    {
                                                                        item.importance
                                                                    }{" "}
                                                                    importance
                                                                </span>
                                                            </div>
                                                        </article>
                                                    ),
                                                )}
                                            </div>
                                        ) : (
                                            <p className="rounded-xl bg-emerald-50 p-4 text-sm leading-6 text-emerald-800">
                                                All identified requirements have
                                                supporting evidence in your
                                                resume. Review the evidence to
                                                ensure it accurately represents
                                                your experience.
                                            </p>
                                        )}
                                    </div>

                                    {/* Strengths */}
                                    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
                                        <h2 className="text-xl font-bold text-slate-900">
                                            Key Strengths
                                        </h2>
                                        <p className="mt-1 text-sm text-slate-500">
                                            Positive aspects identified during
                                            the analysis.
                                        </p>

                                        {analysis.strengths.length > 0 ? (
                                            <ul className="mt-5 space-y-3">
                                                {analysis.strengths.map(
                                                    (strength, index) => (
                                                        <li
                                                            key={`${strength}-${index}`}
                                                            className="flex items-start gap-3 text-sm leading-6 text-slate-700"
                                                        >
                                                            <span className="mt-0.5 text-emerald-600">
                                                                ✓
                                                            </span>
                                                            <span>
                                                                {strength}
                                                            </span>
                                                        </li>
                                                    ),
                                                )}
                                            </ul>
                                        ) : (
                                            <p className="mt-4 text-sm text-slate-500">
                                                No additional strengths were
                                                returned.
                                            </p>
                                        )}
                                    </div>

                                    {/* Recommendations */}
                                    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
                                        <h2 className="text-xl font-bold text-slate-900">
                                            Recommended Improvements
                                        </h2>
                                        <p className="mt-1 text-sm text-slate-500">
                                            Practical steps to improve how your
                                            resume demonstrates your
                                            suitability.
                                        </p>

                                        {analysis.improvements.length > 0 ? (
                                            <div className="mt-5 space-y-4">
                                                {analysis.improvements.map(
                                                    (item, index) => (
                                                        <article
                                                            key={`${item.recommendation}-${index}`}
                                                            className="border-l-4 border-blue-500 pl-4"
                                                        >
                                                            <h3 className="font-semibold text-slate-900">
                                                                {
                                                                    item.recommendation
                                                                }
                                                            </h3>
                                                            <p className="mt-1 text-sm leading-6 text-slate-600">
                                                                {item.reason}
                                                            </p>
                                                        </article>
                                                    ),
                                                )}
                                            </div>
                                        ) : (
                                            <p className="mt-4 text-sm text-slate-500">
                                                No additional recommendations
                                                were returned.
                                            </p>
                                        )}

                                        {analysis.keyword_suggestions.length >
                                            0 && (
                                            <div className="mt-7 border-t border-slate-100 pt-5">
                                                <h3 className="font-semibold text-slate-900">
                                                    Suggested Keywords
                                                </h3>
                                                <p className="mt-1 text-sm text-slate-500">
                                                    Use these only when they
                                                    accurately reflect your
                                                    skills and experience.
                                                </p>

                                                <div className="mt-3 flex flex-wrap gap-2">
                                                    {analysis.keyword_suggestions.map(
                                                        (keyword, index) => (
                                                            <span
                                                                key={`${keyword}-${index}`}
                                                                className="rounded-lg border border-blue-100 bg-blue-50 px-3 py-1.5 text-sm font-medium text-blue-700"
                                                            >
                                                                {keyword}
                                                            </span>
                                                        ),
                                                    )}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </>
                            );
                        })()}
                    </section>
                )}

                <footer className="mt-12 text-center text-sm text-slate-500">
                    Built to help you prepare smarter for your next opportunity.
                    <p className="mt-1 italic text-slate-400 text-xs">
                        *Disclaimer: This is a demo application thus cannot guarantee accuracy to 100%. Do not rely solely on the analysis results for critical decisions. Always verify information independently.
                    </p>
                </footer>
            </section>
        </main>
    );
}
