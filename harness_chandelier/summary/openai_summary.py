"""
summary/openai_summary.py

Dominant topic summarization using OpenAI API.
Requires: pip install openai python-dotenv
Set OPENAI_API_KEY in your .env file or environment variables.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def summarize_texts(text_list: list[str], max_tokens: int = 500, language: str = "en") -> str:
    """
    Summarize a list of messages into a single sentence using OpenAI.

    Args:
        text_list (list[str]): List of messages to summarize (English, Korean, or mixed)
        max_tokens (int): Maximum length of the summary response
        language (str): Output language for the summary ('en' or 'ko')

    Returns:
        str: A one-sentence summary of the dominant topic
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    texts_combined = "\n\n".join([f"[{i+1}] {text}" for i, text in enumerate(text_list)])

    if language == "ko":
        prompt = (
            f"The following {len(text_list)} messages represent the dominant topic "
            f"in a long AI conversation — the subject the user kept returning to.\n"
            f"Summarize the user's core interest in ONE sentence written in Korean.\n\n"
            f"Do NOT use markdown formatting such as ** or ## in your response.\n\n"
            f"Messages:\n{texts_combined}"
        )
    else:
        prompt = (
            f"The following {len(text_list)} messages represent the dominant topic "
            f"in a long AI conversation — the subject the user kept returning to.\n"
            f"Summarize the user's core interest in ONE sentence written in English.\n\n"
            f"Do NOT use markdown formatting such as ** or ## in your response.\n\n"
            f"Messages:\n{texts_combined}"
        )

    response = client.responses.create(
        model="gpt-4o-mini",  # you need a permission for this model, but it works well for summarization tasks
        input=prompt,
        temperature=0.3,
        max_output_tokens=max_tokens,
    )

    return response.output[0].content[0].text.strip()


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
    print("=== Summary (OpenAI) [EN] ===\n")
    print(summary_en)

    summary_ko = summarize_texts(sample_texts, language="ko")
    print("\n=== Summary (OpenAI) [KO] ===\n")
    print(summary_ko)
