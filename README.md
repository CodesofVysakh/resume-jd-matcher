# Resume-JD Matcher

An AI-powered Resume–Job Description Matcher that analyzes a candidate's resume against a job description and provides an explainable assessment of how well the candidate matches the role.

The project is built with a **Django REST Framework backend** and a **Next.js frontend**, with an LLM used for structured job-requirement extraction and resume evaluation.

---

## Features

- Upload a resume in PDF format
- Extract readable text from PDF resumes
- Provide a job description for analysis
- Extract structured requirements from the job description
- Categorize requirements into:
  - Skills
  - Experience
  - Qualifications
  - Other
- Assign requirement importance:
  - High
  - Medium
  - Low
- Evaluate resume evidence against individual job requirements
- Classify requirements as:
  - Demonstrated
  - Partial
  - Unclear
- Identify:
  - Matched skills
  - Missing skills
  - Strengths
  - Areas for improvement
  - Keyword suggestions
- Calculate an overall match score
- Cache analysis results to avoid unnecessary repeated LLM requests
- Validate uploaded files and API input
- Provide a backend health-check endpoint
- Responsive web interface

---

## Architecture

```text
resume-jd-matcher/
│
├── backend/
│   ├── matcher/
│   │   ├── services.py
│   │   ├── views.py
│   │   ├── urls.py
│   │   └── ...
│   │
│   ├── manage.py
│   ├── requirements.txt
│   ├── .env
│   └── ...
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   └── ...
│   ├── components/
│   ├── package.json
│   └── ...
│
└── README.md
```

### High-level flow

```text
                ┌─────────────────────┐
                │      Next.js UI      │
                │      (Frontend)      │
                └──────────┬──────────┘
                           │
                           │ Multipart API request
                           ▼
                ┌─────────────────────┐
                │    Django REST API   │
                │      (Backend)       │
                └──────────┬──────────┘
                           │
                    Extract PDF text
                           │
                           ▼
                ┌─────────────────────┐
                │  JD Requirement      │
                │  Extraction          │
                └──────────┬──────────┘
                           │
                    Structured JD
                    requirements
                           │
                           ▼
                ┌─────────────────────┐
                │ Resume Evaluation   │
                │ against requirements│
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Match Score +       │
                │ Explainable Results │
                └─────────────────────┘
```

---

## Tech Stack

### Frontend

- Next.js
- React
- TypeScript
- ESLint
- Responsive UI

### Backend

- Python
- Django
- Django REST Framework
- PyMuPDF
- Groq API
- LLM-based structured analysis

### Development

- Git
- npm
- Python virtual environment
- REST API architecture

---

## AI Analysis Approach

The matcher is designed around an important principle:

> A match score should be accompanied by evidence explaining why the candidate matches or does not match the requirements.

Instead of relying only on semantic similarity, the system breaks the analysis into individual job requirements.

### Stage 1 — Job Description Requirement Extraction

The job description is first analyzed independently.

Example:

```json
[
  {
    "requirement": "React.js",
    "category": "skill",
    "importance": "high"
  },
  {
    "requirement": "Node.js",
    "category": "skill",
    "importance": "high"
  },
  {
    "requirement": "REST API integration",
    "category": "skill",
    "importance": "high"
  }
]
```

These requirements form the structured representation of the job description.

### Stage 2 — Resume Evaluation

The resume is then evaluated against the extracted requirements.

For each requirement, the system determines the available evidence and classifies it accordingly.

Example:

```json
{
  "requirement": "React.js",
  "category": "skill",
  "importance": "high",
  "status": "demonstrated",
  "evidence": "Built responsive web applications using React.js."
}
```

Possible statuses include:

- `demonstrated` — clear evidence exists in the resume
- `partial` — related or incomplete evidence exists
- `unclear` — sufficient evidence cannot be established from the resume

This approach makes the analysis more transparent than presenting a score alone.

---

## Requirement Normalization

LLMs can express the same requirement in slightly different ways.

For example:

```text
HTML5
HTML5 development
HTML5 and modern web standards
```

The backend therefore performs normalization before requirements are used for evaluation.

Normalization helps reduce differences in:

- Capitalization
- Formatting
- Common wording variations
- Duplicate or overlapping requirements
- Representation of related technologies

The goal is to provide a consistent internal representation while preserving the actual meaning of the job requirement.

---

## Match Score

The application calculates an overall match score based on the structured requirement evaluation.

The score is intended as an **assessment signal**, not an objective measurement of a candidate's actual ability.

For example, a resume may fail to mention a technology even though the candidate has experience with it.

Therefore, the detailed requirement-level evidence is more important than the percentage alone.

---

## Caching

The application includes caching for repeated analysis requests.

Caching helps:

- Reduce unnecessary LLM API calls
- Reduce API consumption
- Improve response times for repeated requests
- Avoid repeatedly processing identical analysis inputs

The cache is implemented on the backend and is based on the analysis input rather than simply caching an arbitrary result.

---

## Backend API

### Health Check

```http
GET /api/health/
```

Example response:

```json
{
  "status": "success",
  "message": "Resume-JD Matcher API is running",
  "version": "1.0.0"
}
```

### Resume Analysis

```http
POST /api/analyze/
```

The endpoint accepts:

- `resume` — PDF file
- `job_description` — job description text

Example request:

```text
multipart/form-data

resume: resume.pdf
job_description: <job description>
```

---

## Input Validation

The backend validates uploaded resumes before processing.

Current validation includes:

- PDF file type
- Maximum file size
- Maximum number of pages
- Password-protected PDF detection
- Minimum readable text requirement
- Job description presence
- Job description length

This prevents invalid input from unnecessarily reaching the AI analysis layer.

---

## Local Development

### Prerequisites

Make sure the following are installed:

- Python 3.12+
- Node.js
- npm
- Git

---

## Backend Setup

Navigate to the backend:

```bash
cd backend
```

Create a virtual environment:

```bash
python3.12 -m venv venv
```

Activate it:

### macOS / Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Run Django checks:

```bash
python manage.py check
```

Start the backend:

```bash
python manage.py runserver
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

---

## Frontend Setup

Open another terminal:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:3000
```

---

## Code Quality Checks

### Frontend linting

```bash
npm run lint
```

### TypeScript type checking

```bash
npx tsc --noEmit
```

`tsc --noEmit` runs the TypeScript compiler's type checking without generating JavaScript output.

Both checks should pass before committing frontend changes.

---

## Environment Variables

Do not commit API keys or other secrets to Git.

Example:

```env
GROQ_API_KEY=your_api_key
```

The `.env` file should be included in `.gitignore`.

A safe approach is to provide an example file such as:

```text
.env.example
```

containing:

```env
GROQ_API_KEY=
```

---

## Error Handling

The backend handles failures from several layers, including:

- Invalid PDF files
- Unreadable/scanned PDFs
- Invalid job descriptions
- LLM API failures
- Invalid LLM responses
- Malformed JSON responses
- Unexpected analysis failures

The API returns appropriate HTTP responses rather than exposing internal implementation details to the frontend.

---

## Project Design Principles

### 1. Separate extraction from evaluation

The job description should first be converted into structured requirements before evaluating the resume.

```text
Job Description
      ↓
Requirement Extraction
      ↓
Requirement Normalization
      ↓
Resume Evaluation
      ↓
Score + Evidence
```

This makes the system easier to test and reason about.

### 2. Evidence over score

The percentage should not be treated as absolute truth.

The system should explain:

```text
Requirement
    ↓
Candidate evidence
    ↓
Status
    ↓
Contribution to overall assessment
```

### 3. Validate LLM output

LLM responses are treated as untrusted structured data.

The backend validates:

- JSON structure
- Required fields
- Allowed categories
- Allowed importance values
- Requirement count
- Expected data types

### 4. Keep deterministic logic outside the LLM

Where possible, business logic such as:

- Validation
- Normalization
- Deduplication
- Score calculation

should be handled by application code rather than relying entirely on the LLM.

---

## Testing Strategy

The project can be tested at multiple levels.

### Requirement extraction

Use a fixed test job description and verify that:

- Requirements are returned
- Required fields exist
- Categories are valid
- Importance values are valid
- Duplicate/overlapping requirements are handled appropriately

### Normalization

Test that equivalent requirement representations normalize to the same internal representation.

### API

Test:

- Valid PDF uploads
- Invalid file types
- Oversized files
- Empty job descriptions
- Invalid PDFs
- LLM/API failures

### Frontend

Run:

```bash
npm run lint
```

and:

```bash
npx tsc --noEmit
```

---

## Future Improvements

Potential future improvements include:

- More robust requirement taxonomy
- Better skill/entity normalization
- Automated regression test datasets
- Embedding-based retrieval for larger requirement sets
- Improved evaluation calibration
- Background processing for large resumes
- Persistent analysis history
- Authentication and user accounts
- Production-grade observability and logging
- Automated evaluation benchmarks for LLM responses
- Support for additional document formats
- More sophisticated evidence extraction

---

## Disclaimer

ResumeMatch provides an AI-assisted assessment based on the information available in a resume and job description.

A resume may not contain every skill or experience a candidate actually possesses. Therefore, the generated match score should be treated as an informational assessment rather than an objective measure of candidate suitability.

The detailed requirement-level evidence should be considered alongside the overall score.

---

## License

This project is currently intended for personal/portfolio use.
