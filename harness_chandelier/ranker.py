import cudf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List

from .topic import extract_topics
from .weights import calculate_weighted_score, scale_wgt, compute_delta_time, normalize_wgt
from .graph import run_pagerank
from .lang import get_lang_defaults


def generate_realistic_timestamps(n_messages: int, base_time: datetime = None) -> List[datetime]:
    if base_time is None:
        base_time = datetime.now()
    timestamps = [base_time]
    for _ in range(n_messages - 1):
        r = np.random.random()
        if r < 0.5:
            gap = np.random.randint(5, 60)
        elif r < 0.8:
            gap = np.random.randint(60, 300)
        else:
            gap = np.random.randint(600, 3600)
        timestamps.append(timestamps[-1] + timedelta(seconds=int(gap)))
    return timestamps


def _build_prompt(messages: List[str], lang: str) -> str:
    if lang == "ko":
        return (
            "다음은 긴 AI 챗봇 대화에서 사용자가 가장 많이 돌아온 핵심 주제의 메시지들입니다.\n"
            "이 메시지들을 바탕으로 사용자의 핵심 관심사를 한 문장으로 요약해주세요.\n\n"
            "메시지:\n" + "\n".join(f"- {m}" for m in messages)
        )
    else:
        return (
            "The following messages represent the dominant topic in a long AI conversation "
            "— the subject the user kept returning to.\n"
            "Summarize the user's core interest in one sentence.\n\n"
            "Messages:\n" + "\n".join(f"- {m}" for m in messages)
        )


def _call_anthropic(prompt: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def _call_openai(prompt: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def _call_gemini(prompt: str, api_key: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


def _call_grok(prompt: str, api_key: str) -> str:
    # Grok uses OpenAI-compatible API
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )
    response = client.chat.completions.create(
        model="grok-beta",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


_LLM_CALLERS = {
    "anthropic": _call_anthropic,
    "openai":    _call_openai,
    "gemini":    _call_gemini,
    "grok":      _call_grok,
}


def _summarize_with_llm(
    messages: List[str],
    api_key: str,
    api_provider: str = "anthropic",
    lang: str = "en"
) -> str:
    """
    Summarize the dominant topic messages using an LLM API.

    Args:
        messages: Messages classified as the main topic.
        api_key: API key for the LLM provider.
        api_provider: "anthropic", "openai", "gemini", or "grok".
        lang: Language code for the prompt. "en" or "ko".

    Returns:
        A one-sentence summary of the dominant topic.
    """
    caller = _LLM_CALLERS.get(api_provider)
    if caller is None:
        raise ValueError(
            f"Unsupported api_provider: '{api_provider}'. "
            f"Choose from: {list(_LLM_CALLERS.keys())}"
        )

    prompt = _build_prompt(messages, lang)

    try:
        return caller(prompt, api_key)
    except Exception as e:
        return f"[Summary unavailable: {e}]"


@dataclass
class RankerResult:
    topic_labels: list
    main_topic: int
    main_topic_keywords: list           # top keywords of the main topic
    main_topic_messages: list           # actual messages classified as main topic
    main_topic_summary: Optional[str]   # LLM summary (None if no api_key provided)
    pagerank: pd.DataFrame
    edge_df: pd.DataFrame


class HarnessChandelier:
    """
    Weight Guidelines:
        delta_time: +0.2  -> Topics user keeps returning to (true intent)
        delta_time: -0.2  -> Frequently switching topics (noise/blockers)

    Language Notes:
        lang="en" -> English stopwords, n_neighbors defaults to 3
        lang="ko" -> Korean stopwords + kiwipiepy tokenizer, n_neighbors defaults to 5
                     English text in Korean conversations is still supported.

    LLM Summary (optional):
        Provide api_key and api_provider to enable automatic topic summarization.
        Supported providers: "anthropic", "openai", "gemini", "grok"
        If api_key is not provided, main_topic_summary will be None.

    Example:
        # No LLM summary
        ranker = HarnessChandelier(lang="ko")

        # With Anthropic
        ranker = HarnessChandelier(lang="ko", api_key="sk-ant-...", api_provider="anthropic")

        # With OpenAI
        ranker = HarnessChandelier(lang="ko", api_key="sk-...", api_provider="openai")

        # With Gemini
        ranker = HarnessChandelier(lang="ko", api_key="AI...", api_provider="gemini")

        # With Grok
        ranker = HarnessChandelier(lang="ko", api_key="xai-...", api_provider="grok")
    """

    def __init__(
        self,
        lang: str = "en",
        weights: Optional[dict] = None,
        scale_factor: int = 1000,
        scaler: str = "scale_and_round",
        n_neighbors: Optional[int] = None,
        min_cluster_size: int = 3,
        base_time: Optional[datetime] = None,
        random_seed: int = 42,
        api_key: Optional[str] = None,      # LLM API key (optional)
        api_provider: str = "anthropic",    # "anthropic", "openai", "gemini", "grok"
    ):
        self.lang = lang
        self.weights = weights or {"delta_time": +0.2, "transition_count": 1.0}
        self.scale_factor = scale_factor
        self.scaler = scaler
        self.min_cluster_size = min_cluster_size
        self.base_time = base_time or datetime.now()
        self.random_seed = random_seed
        self.api_key = api_key
        self.api_provider = api_provider
        self.topic_model = None
        self.messages = []
        self.timestamps = []

        # n_neighbors: if the user specifies a value, use it; otherwise, use the lang default
        lang_defaults = get_lang_defaults(lang)
        self.n_neighbors = n_neighbors if n_neighbors is not None else lang_defaults["n_neighbors"]

    def add_message(self, message: str, timestamp: datetime = None):
        self.messages.append(message)
        self.timestamps.append(timestamp or datetime.now())

        if len(self.messages) >= 5:
            return self.fit(self.messages, self.timestamps)
        return None

    def fit(self, messages: list, timestamps: Optional[List[datetime]] = None) -> RankerResult:
        np.random.seed(self.random_seed)

        topics, probs, topic_model = extract_topics(
            messages,
            n_neighbors=self.n_neighbors,
            min_cluster_size=self.min_cluster_size,
            lang=self.lang
        )

        valid_idx = [i for i, t in enumerate(topics) if t != -1]
        if len(valid_idx) < 3:
            return None

        if timestamps is None:
            timestamps = generate_realistic_timestamps(
                n_messages=len(messages),
                base_time=self.base_time
            )

        if len(timestamps) != len(messages):
            raise ValueError(
                f"timestamps length ({len(timestamps)}) must match messages length ({len(messages)})"
            )

        edges = []
        for i in range(len(topics) - 1):
            if topics[i] == -1 or topics[i + 1] == -1:
                continue
            edges.append({
                'src': topics[i],
                'dst': topics[i + 1],
                'timestamp': timestamps[i]
            })

        pdf = pd.DataFrame(edges)
        pdf['timestamp'] = pd.to_datetime(pdf['timestamp'])
        pdf = compute_delta_time(pdf)
        pdf['transition_count'] = pdf.groupby(['src', 'dst'])['src'].transform('count')
        pdf = calculate_weighted_score(pdf, self.weights)
        pdf = normalize_wgt(pdf)
        pdf = scale_wgt(pdf, self.scaler, self.scale_factor)

        gdf = cudf.from_pandas(pdf[['src', 'dst', 'wgt']].astype('int32'))
        result = run_pagerank(gdf)

        # extract the main topic keywords
        main_keywords = []
        if topic_model is not None:
            main_keywords = [w for w, _ in topic_model.get_topic(result.main_topic)[:5]]

        # extract the main topic messages
        main_topic_messages = [
            messages[i] for i, t in enumerate(topics)
            if t == result.main_topic
        ]

        # LLM summmarization (when api_key is provided and there are messages in the main topic)
        main_topic_summary = None
        if self.api_key and main_topic_messages:
            main_topic_summary = _summarize_with_llm(
                messages=main_topic_messages,
                api_key=self.api_key,
                api_provider=self.api_provider,
                lang=self.lang
            )

        return RankerResult(
            topic_labels=topics,
            main_topic=result.main_topic,
            main_topic_keywords=main_keywords,
            main_topic_messages=main_topic_messages,
            main_topic_summary=main_topic_summary,
            pagerank=result.pagerank,
            edge_df=pdf
        )
