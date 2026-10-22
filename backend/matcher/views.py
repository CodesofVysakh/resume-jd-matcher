
import logging

import groq
import pymupdf

from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from .services import (
    RequirementExtractionError,
    analyze_resume,
    get_job_requirements,
)

logger = logging.getLogger(__name__)


@api_view(["GET"])
def health_check(request):
    return Response({
        "status": "success",
        "message": "Resume-JD Matcher API is running",
        "version": "1.0.0",
    })


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def analyze_resume_view(request):
    resume = request.FILES.get("resume")
    job_description = request.data.get("job_description", "")

    if not resume:
        return Response(
            {"error": "Please upload a resume PDF."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if not resume.name.lower().endswith(".pdf"):
        return Response(
            {"error": "Only PDF files are supported."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if resume.size > 5 * 1024 * 1024:
        return Response(
            {"error": "The PDF must be 5 MB or smaller."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if not isinstance(job_description, str) or not job_description.strip():
        return Response(
            {"error": "Please provide a job description."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if len(job_description) > 20000:
        return Response(
            {"error": "The job description is too long."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        pdf_bytes = resume.read()

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.needs_pass:
                return Response(
                    {"error": "Password-protected PDFs are not supported."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if len(document) == 0 or len(document) > 20:
                return Response(
                    {"error": "The PDF must contain 1–20 pages."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            resume_text = "\n".join(
                page.get_text() for page in document
            ).strip()

    except Exception:
        return Response(
            {"error": "Unable to read this PDF. Please check the file."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if len(resume_text) < 50:
        return Response(
            {
                "error": (
                    "Insufficient readable text was found. "
                    "The PDF may be scanned or image-based."
                )
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        # Step 1: the job requirements. The same job description always gives
        # the same list (see get_job_requirements).
        requirements = get_job_requirements(job_description)

        # Step 2: evaluate the resume against exactly those requirements.
        result = analyze_resume(
            resume_text=resume_text,
            job_description=job_description.strip(),
            requirements=requirements,
        )
    except RequirementExtractionError as e:
        # Retrying will not help: the job description itself needs changing.
        return Response(
            {"error": str(e)},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except groq.RateLimitError:
        return Response(
            {
                "error": (
                    "The AI service is busy right now. "
                    "Please try again in a minute."
                )
            },
            status=status.HTTP_429_TOO_MANY_REQUESTS,
        )
    except Exception as e:
        logger.exception(
            "Resume analysis error: %s: %s", type(e).__name__, e
        )
        return Response(
            {
                "error": (
                    "Resume analysis failed. Please try again "
                    "or check the backend configuration."
                )
            },
            status=status.HTTP_502_BAD_GATEWAY,
        )

    return Response({
        "status": "success",
        "filename": resume.name,
        "analysis": result,
    })
