"""
Reusable module: website summarizer using Ollama (OpenAI-compatible endpoint).

This wraps the core logic so it can be imported from notebooks, scripts, or other modules.
"""

import os
from typing import List, Dict

from openai import OpenAI
from scraper import fetch_website_contents

# Ollama OpenAI-compatible endpoint
OLLAMA_BASE_URL: str = "http://localhost:11434/v1"

# Configure model via environment variable, with a sensible default.
# Example: OLLAMA_MODEL=llama3.2:1b for smaller machines.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

# Dummy key is fine – Ollama ignores it but the client requires one.
_ollama_client: OpenAI = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

# Prompts reused across calls (mirroring the Day 1 structure).
SYSTEM_PROMPT: str = """
You are a snarky assistant that analyzes the contents of a website,
and provides a short, snarky, humorous summary, ignoring text that might be navigation related.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""

USER_PROMPT_PREFIX: str = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.

"""


def messages_for(website: str) -> List[Dict[str, str]]:
    """
    Create the messages list for the LLM, in OpenAI / Ollama chat format.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_PREFIX + website},
    ]


def summarize(url: str) -> str:
    """
    Fetch and summarize a website using the local Ollama model.

    :param url: URL of the website to summarize.
    :return: Markdown-formatted summary text from the model.
    """
    website = fetch_website_contents(url)
    response = _ollama_client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=messages_for(website),
    )
    return response.choices[0].message.content


__all__ = [
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "SYSTEM_PROMPT",
    "USER_PROMPT_PREFIX",
    "messages_for",
    "summarize",
]

