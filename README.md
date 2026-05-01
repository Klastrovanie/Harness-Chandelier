# Harness-Chandelier
Harness the Chandelier — connect to the center of your conversation.

**First Public Release:** 2026-04-11  
**Last Updated:** 2026-05-01

---

## What is this?

In real-world AI agent interactions, users naturally drift across multiple topics —
then return to what they originally wanted.

Harness-Chandelier doesn't try to prevent drift.
It **tracks it** — and finds the topic the user kept coming back to.

Like a chandelier at the center of a hall, the dominant topic stays fixed
no matter how much the conversation moves around it.

---

## Why does this matter?

Anthropic's Harness Engineering solves **"how the AI maintains context"**.  
Harness-Chandelier solves **"what the user actually wants"**.

**Harness-Chandelier focuses on continuously discovering the user’s persistently repetitive intent** — the topics they keep returning to over long periods of time, even when the conversation drifts wildly.

They are complementary — not competing.

> While harness design ensures the AI agent maintains context across sessions,  
> Harness-Chandelier ensures the agent always knows what the user **actually wants**.

---

## How it works

1. **BERTopic** + cuML UMAP/HDBSCAN — extract topics from each message (GPU-accelerated, no generative LLM required)
2. **Topic Transition Graph** — model topic flow as `src → dst` edges with real timestamps
3. **cuGraph PageRank** — rank topics by importance (most central = dominant)

### Weight Guidelines

| Weight | Behavior | Use Case |
|--------|----------|----------|
| `delta_time: +0.2` | Topics user keeps returning to | **Detect true user intent** |
| `delta_time: -0.2` | Frequently switching topics | Detect noise / interruptions |

> Short, context-free messages tend to become outliers (Topic -1) in BERTopic,  
> which Harness-Chandelier naturally assigns the lowest PageRank —  
> effectively filtering noise from the dominant topic detection.

---

## Quick Start

```bash
conda env create -f environment.yml
conda activate rapids
```

```python
from harness_chandelier import HarnessChandelier
# English
ranker = HarnessChandelier(
    lang="en",
    weights={"delta_time": +0.2}
)

# Korean (English mixed conversations also supported)
ranker = HarnessChandelier(
    lang="ko",
    weights={"delta_time": +0.2}
)
```

### Option 1: Batch (full conversation at once)
```python
result = ranker.fit(messages, timestamps=real_timestamps)
 
print(f"Main Topic: {result.main_topic}")
print(f"Keywords: {result.main_topic_keywords}")
print(f"Messages: {result.main_topic_messages}")
```

### Option 2: AI API Summary
```python
from harness_chandelier.summary import summarize
 
# Summarize dominant topic messages with your preferred AI provider
summary = summarize(
    result.main_topic_messages,
    provider="anthropic",   # "anthropic", "openai", "gemini", "grok"
    language="en"           # "en" or "ko"
)
print(summary)
```

### Option 3: Real-time (message by message)
```python
for message in live_messages:
    result = ranker.add_message(message)
 
    if result:  # returns result every 5+ messages
        print(f"Current Dominant Topic: {result.main_topic}")
```
---

Set your API key in `.env` for AI summary option:
 
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AI...
XAI_API_KEY=xai-...
```

---
 
## Language Support
 
| Language | `lang=` | Tokenizer | Stopwords |
|----------|---------|-----------|-----------|
| English | `"en"` | BERTopic default | English |
| Korean | `"ko"` | kiwipiepy (LGPL v3) | Korean |
| Korean + English mixed | `"ko"` | kiwipiepy | Korean |
| Other languages | `"en"` | BERTopic default | English |
 
The embedding model `intfloat/multilingual-e5-base` supports 100+ languages regardless of `lang` setting.
The `lang` parameter only affects stopword filtering and tokenization for keyword extraction.
 
---


## Example Datasets

Three real-world conversation scenarios included:

| Example | Scenario | Pattern |
|---------|----------|---------|
| `example-coder.ipynb` | AI repeatedly misses user's design specs | User keeps restating original intent |
| `example-mixed-topics.ipynb` | User jumps between travel, coding, life advice | User returns to dominant topic |
| `example-mixed-korean-topics.ipynb` | Travel / Coding / Life advice (batch) | Korean |

> **Note:** All demo codes were performed on AWS g4dn.xlarge  
> (NVIDIA T4 16GB GPU, the most affordable GPU instance on AWS).  
> A CUDA-compatible GPU is required to run this project.

---

## Prior Art Declaration

This repository was first made publicly available on **April 11, 2026**.

It establishes definitive prior art for the following technical contribution 
worldwide (United States, Europe, China, Korea, and all other jurisdictions 
under the Paris Convention and PCT):

**"Reverse Topic-Drift PageRank for Long-Term User Intent Tracking 
in Harness-Style Long-Running AI Agents"**

Specifically, this work discloses and implements for the first time:

- Topic extraction from user messages using BERTopic (GPU-accelerated with cuML)
- Construction of a directed time-weighted transition graph between consecutive topics
- Edge weighting that gives positive reward to `delta_time` (returning to the same 
  topic after a time gap = strong signal of true user intent) combined with 
  transition count
- Application of cuGraph PageRank on this directed graph to rank the most 
  persistent user topic (`main_topic`)
- Use of this mechanism as a **user-side complement** to Anthropic's Harness 
  Engineering (agent-side context management)

By publishing this repository with full enabling disclosure 
(code + methodology + mathematical formulation), we dedicate this invention 
to the public domain as prior art, **preventing future patenting of this 
specific combination by any party**.

---

## License

AGPL v3 — free for research and non-commercial use.  
The complete algorithm, source code, weighting formulas, pseudocode, and working implementation are provided openly under AGPL v3.
For commercial licensing, leave a message on [Discussions](../../discussions).

- intfloat/multilingual-e5-base — MIT License (Microsoft Research)
- kiwipiepy — LGPL v3 (commercial use allowed)
- BERTopic — MIT License
- cuML, cuGraph — Apache 2.0 (NVIDIA RAPIDS)
---

## Disclaimer

Harness-Chandelier is not affiliated with, endorsed by, or related to Anthropic.  
"Harness Engineering" is a term used in Anthropic's engineering blog.  
This project addresses a complementary problem in the same ecosystem.

The example conversation datasets included in this repository reference "Figma"  
as a realistic user scenario. Figma® is a registered trademark of Figma, Inc.  
This project is not affiliated with, endorsed by, or related to Figma, Inc. in any way.

The example conversation datasets also reference "ChatGPT" as a realistic user scenario.  
ChatGPT® is a registered trademark of OpenAI. This project is not affiliated with,  
endorsed by, or related to OpenAI in any way.

---

## Copyright

Copyright © 2026 Klastrovanie Co., Ltd. All rights reserved.

