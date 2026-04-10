# Harness-Chandelier
Harness the Chandelier — connect to the center of your conversation.

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

ranker = HarnessChandelier(
    weights={"delta_time": +0.2}
)

# Option 1: auto-generate realistic timestamps
result = ranker.fit(messages)

# Option 2: provide real timestamps
result = ranker.fit(messages, timestamps=real_timestamps)

print(result.main_topic)    # dominant topic
print(result.pagerank)      # topic importance scores
print(result.topic_labels)  # topic per message
```
---

## Example Datasets

Three real-world conversation scenarios included:

| Example | Scenario | Pattern |
|---------|----------|---------|
| `example-coder.ipynb` | AI repeatedly misses user's design specs | User keeps restating original intent |
| `example-mixed-topics.ipynb` | User jumps between travel, coding, life advice | User returns to dominant topic |

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

---

## Disclaimer

Harness-Chandelier is not affiliated with, endorsed by, or related to Anthropic.  
"Harness Engineering" is a term used in Anthropic's engineering blog.  
This project addresses a complementary problem in the same ecosystem.

The example conversation datasets included in this repository reference "Figma"  
as a realistic user scenario. Figma® is a registered trademark of Figma, Inc.  
This project is not affiliated with, endorsed by, or related to Figma, Inc. in any way.

---

## Copyright

Copyright © 2026 Klastrovanie Co., Ltd. All rights reserved.

