"""
summary/grok_summary.py

Dominant topic summarization using xAI Grok API.
Requires: pip install openai python-dotenv
Set XAI_API_KEY in your .env file or environment variables.

Note: Grok uses an OpenAI-compatible API endpoint (https://api.x.ai/v1).
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def summarize_texts(text_list: list[str], max_tokens: int = 500, language: str = "en") -> str:
    """
    Summarize a list of messages into a single sentence using xAI Grok.

    Args:
        text_list (list[str]): List of messages to summarize (English, Korean, or mixed)
        max_tokens (int): Maximum length of the summary response
        language (str): Output language for the summary ('en' or 'ko')

    Returns:
        str: A one-sentence summary of the dominant topic
    """
    # Grok uses OpenAI-compatible API with a custom base URL
    client = OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1"
    )

    texts_combined = "\n\n".join([f"[{i+1}] {text}" for i, text in enumerate(text_list)])

    system_prompt = (
        "You are an expert at analyzing conversation patterns. "
        "Your task is to identify and summarize the dominant topic "
        "that a user keeps returning to in a long AI conversation."
    )

    if language == "ko":
        user_prompt = (
            f"The following {len(text_list)} messages represent the dominant topic "
            f"in a long AI conversation — the subject the user kept returning to.\n"
            f"Summarize the user's core interest in ONE sentence written in Korean.\n\n"
            f"Do NOT use markdown formatting such as ** or ## in your response.\n\n"
            f"Messages:\n{texts_combined}"
        )
    else:
        user_prompt = (
            f"The following {len(text_list)} messages represent the dominant topic "
            f"in a long AI conversation — the subject the user kept returning to.\n"
            f"Summarize the user's core interest in ONE sentence written in English.\n\n"
            f"Do NOT use markdown formatting such as ** or ## in your response.\n\n"
            f"Messages:\n{texts_combined}"
        )

    response = client.chat.completions.create(
        model="grok-4.3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


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
    print("=== Summary (xAI Grok) [EN] ===\n")
    print(summary_en)

    summary_ko = summarize_texts(sample_texts, language="ko")
    print("\n=== Summary (xAI Grok) [KO] ===\n")
    print(summary_ko)
