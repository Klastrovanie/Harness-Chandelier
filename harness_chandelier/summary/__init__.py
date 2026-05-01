"""
harness_chandelier/summary/__init__.py

Unified summarization interface for Harness-Chandelier.
Supports Anthropic, OpenAI, Gemini, and Grok providers.

Usage:
    from harness_chandelier.summary import summarize

    # English summary with Anthropic Claude
    summarize(messages, provider="anthropic", language="en")

    # Korean summary with Grok
    summarize(messages, provider="grok", language="ko")
"""


def summarize(
    text_list: list[str],
    provider: str = "anthropic",
    language: str = "en",
    **kwargs
) -> str:
    """
    Summarize a list of messages into a single sentence.

    Args:
        text_list (list[str]): Messages to summarize (English, Korean, or mixed)
        provider (str): LLM provider — "anthropic", "openai", "gemini", or "grok"
        language (str): Output language — "en" or "ko"
        **kwargs: Additional arguments passed to the provider (e.g. max_tokens)

    Returns:
        str: A one-sentence summary of the dominant topic

    Raises:
        ValueError: If an unsupported provider is specified
    """
    if provider == "anthropic":
        from .anthropic_summary import summarize_texts
    elif provider == "openai":
        from .openai_summary import summarize_texts
    elif provider == "gemini":
        from .gemini_summary import summarize_texts
    elif provider == "grok":
        from .grok_summary import summarize_texts
    else:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            f"Choose from: 'anthropic', 'openai', 'gemini', 'grok'"
        )

    return summarize_texts(text_list, language=language, **kwargs)
