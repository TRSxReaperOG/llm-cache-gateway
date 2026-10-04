# llm-cache-gateway

[![CI](https://github.com/TRSxReaperOG/llm-cache-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/TRSxReaperOG/llm-cache-gateway/actions/workflows/ci.yml)

## Problem

Every application built on LLM APIs pays for the same thing over and over: near-identical prompts trigger full, billable model calls every single time, even when the "new" question is really just a rewording of something asked minutes (or seconds) ago.

- **Cost**: LLM API calls are billed per token. A support bot, internal tool, or any high-traffic LLM feature ends up paying full price for answers it has effectively already generated.
- **Latency**: Every call waits on a real model response — typically hundreds of milliseconds to a few seconds — even for a question the system has already answered.
- **Reliability**: Every avoidable call also counts against provider rate limits, making outages/throttling more likely exactly when traffic spikes.

Traditional exact-match caching barely helps, because real users rarely type the identical string twice — "What's your refund policy?" and "How do I get a refund?" mean the same thing but look like two different cache keys to a normal cache.

**llm-cache-gateway** sits between an application and its LLM provider as a drop-in proxy. It checks incoming prompts for *semantic* similarity to previously-seen prompts — not exact text match — and serves a cached response when a close-enough match exists, instead of paying for and waiting on a new model call.
