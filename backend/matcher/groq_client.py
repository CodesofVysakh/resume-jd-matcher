import os

from groq import Groq

MODEL_NAME = "openai/gpt-oss-20b"


def get_groq_client():
    # GROQ_API_KEY is the usual name; API_KEY is still accepted because that
    # is what this project originally read (the error below always said
    # GROQ_API_KEY, which was misleading).
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Check your backend .env file."
        )

    return Groq(api_key=api_key)


def test_groq_connection():
    client = get_groq_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Groq connection successful",
            }
        ],
        max_completion_tokens=100,
    )

    return response.choices[0].message.content
