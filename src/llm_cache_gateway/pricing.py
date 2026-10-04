# Rough, illustrative blended price per 1K tokens, USD. NOT exact per-model
# billing — real pricing varies by model within a provider. Good enough for a
# ballpark cost-estimate in request logs, not for actual billing reconciliation.
PRICE_PER_1K_TOKENS_USD = {
    "openai": 0.005,
    "groq": 0.0002,
    "mistral": 0.002,
    "together": 0.0009,
    "anthropic": 0.006,
    "gemini": 0.0005,
    "cohere": 0.0015,
}


def estimate_cost(provider: str, tokens: int) -> float:
    price = PRICE_PER_1K_TOKENS_USD.get(provider, 0.0)
    return (tokens / 1000) * price
