
from io import BytesIO

from django.test import SimpleTestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status


class ResumeMatcherAPITests(SimpleTestCase):
    def setUp(self):
        self.client = APIClient()
        self.url = reverse("analyze-resume")

    def test_missing_resume_returns_400(self):
        response = self.client.post(
            self.url,
            {"job_description": "React developer"},
            format="multipart",
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_400_BAD_REQUEST,
        )
        self.assertIn("error", response.data)

    def test_missing_job_description_returns_400(self):
        response = self.client.post(
            self.url,
            {
                "resume": self.create_pdf_upload(),
                "job_description": "",
            },
            format="multipart",
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_400_BAD_REQUEST,
        )

    def test_non_pdf_upload_returns_400(self):
        response = self.client.post(
            self.url,
            {
                "resume": BytesIO(b"not a PDF"),
                "job_description": "React developer",
            },
            format="multipart",
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_400_BAD_REQUEST,
        )

    def test_health_endpoint_returns_success(self):
        response = self.client.get("/api/health/")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["status"], "success")

    @staticmethod
    def create_pdf_upload():
        # Minimal PDF with readable text, generated for testing.
        import pymupdf

        document = pymupdf.open()
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Experienced React developer with TypeScript and REST APIs. "
            "Built web applications and worked with frontend teams.",
        )

        pdf_bytes = document.tobytes()
        document.close()

        from django.core.files.uploadedfile import SimpleUploadedFile

        return SimpleUploadedFile(
            "resume.pdf",
            pdf_bytes,
            content_type="application/pdf",
        )