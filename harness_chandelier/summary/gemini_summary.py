"""
summary/gemini_summary.py

Dominant topic summarization using Google Gemini API.
Requires: pip install google-genai python-dotenv
Set GOOGLE_API_KEY in your .env file or environment variables.
"""
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()


def summarize_texts(text_list: list[str], language: str = "en") -> str:
    """
    Summarize a list of messages into a single sentence using Google Gemini.

    Args:
        text_list (list[str]): List of messages to summarize (English, Korean, or mixed)
        language (str): Output language for the summary ('en' or 'ko')

    Returns:
        str: A one-sentence summary of the dominant topic
    """
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    texts_combined = "\n\n".join([f"[{i+1}] {text}" for i, text in enumerate(text_list)])

    if language == "ko":
        prompt = (
            f"The following {len(text_list)} messages represent the dominant topic "
            f"in a long AI conversation — the subject the user kept returning to.\n"
            f"Summarize the user's core interest in ONE sentence written in Korean.\n"
            f"Do NOT use markdown formatting such as ** or ## in your response.\n\n"
            f"Messages:\n{texts_combined}"
        )
    else:
        prompt = (
            f"The following {len(text_list)} messages represent the dominant topic "
            f"in a long AI conversation — the subject the user kept returning to.\n"
            f"Summarize the user's core interest in ONE sentence written in English.\n"
            f"Do NOT use markdown formatting such as ** or ## in your response.\n\n"
            f"Messages:\n{texts_combined}"
        )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()


# ==================== Example Usage ====================
if __name__ == "__main__":
    sample_texts = [
        "I've been at this company for 7 years but I feel stuck. No growth.",
        "Back to the job thing. I got a recruiter message today on LinkedIn.",
        "I applied to the startup job. Just did the first interview.",
        "They asked me where I see myself in 5 years. I had no idea what to say.",
        "I got the job offer. They want an answer by Friday.",
    ]

    summary_en = summarize_texts(sample_texts, language="en")
    print("=== Summary (Google Gemini) [EN] ===\n")
    print(summary_en)

    summary_ko = summarize_texts(sample_texts, language="ko")
    print("\n=== Summary (Google Gemini) [KO] ===\n")
    print(summary_ko)
