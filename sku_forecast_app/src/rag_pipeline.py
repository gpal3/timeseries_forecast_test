"""Retail news Retrieval-Augmented Generation (RAG) analysis pipeline.

This module implements a lightweight RAG-style workflow tailored for
detecting retail category trends from external news sources. The pipeline
fetches articles, extracts clean text, chunks the content, scores sentiment
signals, classifies the trend direction, and generates embeddings suitable for
downstream semantic retrieval.

The implementation intentionally avoids heavyweight dependencies so that it can
be used in serverless or batch data pipelines with minimal setup.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import requests
import tldextract
from bs4 import BeautifulSoup
from dateutil import parser as date_parser


LOGGER = logging.getLogger(__name__)


# Sentiment dictionaries provided in the specification.
SENTIMENT_KEYWORDS: Dict[str, Sequence[str]] = {
    "positive": (
        "record sales",
        "strong growth",
        "increase",
        "higher demand",
        "raised guidance",
        "expanding",
    ),
    "negative": (
        "decline",
        "decrease",
        "weak demand",
        "cut guidance",
        "slowdown",
    ),
    "neutral": (
        "flat",
        "stable",
        "mixed",
    ),
}


# Keyword lists for each retail category. The goal is to capture the most
# common mentions related to each vertical without an exhaustive taxonomy.
CATEGORY_KEYWORDS: Dict[str, Sequence[str]] = {
    "Apparel": (
        "apparel",
        "clothing",
        "fashion",
        "garment",
        "outerwear",
    ),
    "Grocery & Food Retail": (
        "grocery",
        "supermarket",
        "food retail",
        "grocer",
        "convenience store",
    ),
    "Consumer Electronics": (
        "consumer electronics",
        "electronics retailer",
        "smartphone",
        "laptop",
        "appliance",
    ),
    "Beauty & Personal Care": (
        "beauty",
        "cosmetics",
        "personal care",
        "skincare",
        "makeup",
    ),
    "Home & Furniture / Home Furnishings": (
        "home furnishings",
        "home furniture",
        "furniture retailer",
        "home decor",
        "interior",
    ),
    "Luxury": (
        "luxury",
        "premium brand",
        "designer label",
        "high-end",
        "luxury goods",
    ),
    "Sporting Goods & Athleisure / Footwear": (
        "sporting goods",
        "athleisure",
        "footwear",
        "sportswear",
        "athletic apparel",
    ),
    "Home Improvement & Gardening": (
        "home improvement",
        "do-it-yourself",
        "hardware store",
        "gardening",
        "lawn care",
    ),
}


@dataclass
class FetchedDocument:
    """Container for fetched document state."""

    source_link: Optional[str]
    raw_html: Optional[str]
    provided_text: Optional[str]
    source_name: str


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def fetch_url(item: str, timeout: float = 10.0) -> FetchedDocument:
    """Fetch article content for a URL or wrap plain text snippets.

    Parameters
    ----------
    item:
        URL string or plain text snippet.
    timeout:
        Timeout applied to HTTP requests.

    Returns
    -------
    FetchedDocument
        Document container with either HTML content (for URLs) or the provided
        text snippet.
    """

    if not item:
        raise ValueError("Input item is empty; unable to fetch content.")

    if _is_url(item):
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; RetailTrendBot/1.0; +https://example.com/bot)"
                )
            }
            response = requests.get(item, headers=headers, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.error("Failed to fetch URL %s: %s", item, exc)
            raise

        domain_info = tldextract.extract(item)
        source_name = ".".join(part for part in (domain_info.domain, domain_info.suffix) if part)
        return FetchedDocument(
            source_link=item,
            raw_html=response.text,
            provided_text=None,
            source_name=source_name or "Unknown Source",
        )

    # Plain text snippet branch.
    return FetchedDocument(
        source_link=None,
        raw_html=None,
        provided_text=item,
        source_name="Provided Snippet",
    )


def extract_main_text(html: str) -> str:
    """Extract the main textual content from the supplied HTML string."""

    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "aside"]):
        tag.decompose()

    text = soup.get_text("\n")
    cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(cleaned_lines)


def extract_published_date(html: str, url: Optional[str] = None) -> Optional[str]:
    """Attempt to extract the publication date from HTML metadata or URL."""

    if not html and not url:
        return None

    soup = BeautifulSoup(html, "html.parser") if html else None
    meta_keys = (
        ("meta", {"property": "article:published_time"}),
        ("meta", {"name": "pubdate"}),
        ("meta", {"name": "publishdate"}),
        ("meta", {"name": "date"}),
        ("meta", {"property": "og:updated_time"}),
        ("time", {}),
    )

    content: Optional[str] = None
    if soup:
        for tag_name, attrs in meta_keys:
            tag = soup.find(tag_name, attrs=attrs)
            if tag and tag.get("content"):
                content = tag["content"].strip()
                break
            if tag_name == "time" and tag and tag.get("datetime"):
                content = tag["datetime"].strip()
                break

    if not content and url:
        # Attempt to infer date from URL patterns such as /2024/05/12/.
        match = re.search(r"/(20\d{2})/(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])", url)
        if match:
            content = "-".join(match.groups())

    if not content:
        return None

    try:
        dt = date_parser.parse(content)
        return dt.date().isoformat()
    except (ValueError, TypeError, OverflowError) as exc:
        LOGGER.debug("Unable to parse publication date '%s': %s", content, exc)
        return None


def chunk_text(text: str, tokens_per_chunk: int = 250) -> List[str]:
    """Chunk text into approximately ``tokens_per_chunk`` word tokens."""

    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    for start in range(0, len(words), tokens_per_chunk):
        chunk_words = words[start : start + tokens_per_chunk]
        chunk = " ".join(chunk_words)
        if chunk:
            chunks.append(chunk)
    return chunks


def _count_keyword_occurrences(text: str, keywords: Iterable[str]) -> int:
    text_lower = text.lower()
    return sum(text_lower.count(keyword) for keyword in keywords)


def score_signals(chunk: str) -> Dict[str, Dict[str, int]]:
    """Score sentiment signals for each category present in the chunk."""

    scores: Dict[str, Dict[str, int]] = {}
    chunk_lower = chunk.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        if not any(keyword in chunk_lower for keyword in keywords):
            continue

        category_scores = {
            sentiment: _count_keyword_occurrences(chunk_lower, phrases)
            for sentiment, phrases in SENTIMENT_KEYWORDS.items()
        }

        if any(category_scores.values()):
            scores[category] = category_scores

    return scores


def classify_trend(sentiment_counts: MutableMapping[str, int]) -> str:
    """Classify the trend direction based on sentiment counts."""

    positive = sentiment_counts.get("positive", 0)
    negative = sentiment_counts.get("negative", 0)

    if positive > negative:
        return "Growing"
    if negative > positive:
        return "Declining"
    return "Stable"


def get_best_excerpt(chunks: Sequence[Tuple[str, Dict[str, int]]], category: str) -> str:
    """Select the chunk with the strongest net positive signal for a category."""

    best_chunk: Optional[str] = None
    best_score = -math.inf

    for chunk, scores in chunks:
        net_score = scores.get("positive", 0) - scores.get("negative", 0)
        neutral_weight = 0.1 * scores.get("neutral", 0)
        score = net_score + neutral_weight
        if score > best_score and any(scores.values()):
            best_score = score
            best_chunk = chunk

    return best_chunk or (chunks[0][0] if chunks else "")


def _hash_to_embedding(text: str, dimensions: int = 64) -> List[float]:
    """Generate a deterministic pseudo-embedding using SHA-256 hashes."""

    if not text:
        return [0.0] * dimensions

    seed = text.encode("utf-8")
    digest = hashlib.sha256(seed).digest()
    buffer = bytearray(digest)
    while len(buffer) < dimensions * 4:
        digest = hashlib.sha256(buffer).digest()
        buffer.extend(digest)

    vector: List[float] = []
    for idx in range(dimensions):
        start = idx * 4
        chunk = buffer[start : start + 4]
        integer = int.from_bytes(chunk, "big", signed=False)
        value = ((integer % 2000) - 1000) / 1000.0
        vector.append(value)
    return vector


def generate_embedding(text: str, api_key: Optional[str] = None) -> List[float]:
    """Generate an embedding vector via OpenAI or a deterministic fallback."""

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _hash_to_embedding(text)

    try:
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": "text-embedding-3-large", "input": text},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not data:
            raise ValueError("No embedding data returned by API.")
        embedding = data[0].get("embedding")
        if not embedding:
            raise ValueError("Embedding payload missing 'embedding' key.")
        return embedding
    except (requests.RequestException, ValueError) as exc:
        LOGGER.warning("Falling back to hash-based embedding: %s", exc)
        return _hash_to_embedding(text)


def _build_reason(category: str, counts: Dict[str, int]) -> str:
    positive = counts.get("positive", 0)
    negative = counts.get("negative", 0)
    neutral = counts.get("neutral", 0)
    return (
        f"Detected {positive} positive, {negative} negative, and {neutral} neutral "
        f"signals for {category.lower()} news segments."
    )


def analyze_retail_categories(items: Iterable[str]) -> List[Dict[str, object]]:
    """Run the full RAG workflow across the supplied URLs or text snippets."""

    results: List[Dict[str, object]] = []

    for item in items:
        try:
            document = fetch_url(item)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error("Skipping item due to fetch error: %s", exc)
            continue

        raw_text = ""
        published_date: Optional[str] = None
        if document.raw_html:
            raw_text = extract_main_text(document.raw_html)
            published_date = extract_published_date(document.raw_html, document.source_link)
        elif document.provided_text:
            raw_text = document.provided_text
            published_date = datetime.utcnow().date().isoformat()

        if not raw_text:
            LOGGER.info("No textual content extracted for item: %s", item)
            continue

        chunks = chunk_text(raw_text)
        if not chunks:
            LOGGER.info("No chunks produced for item: %s", item)
            continue

        category_signals: Dict[str, Dict[str, int]] = {}
        category_chunks: Dict[str, List[Tuple[str, Dict[str, int]]]] = {}

        for chunk in chunks:
            scores = score_signals(chunk)
            for category, counts in scores.items():
                aggregate = category_signals.setdefault(
                    category, {"positive": 0, "negative": 0, "neutral": 0}
                )
                for sentiment in ("positive", "negative", "neutral"):
                    aggregate[sentiment] += counts.get(sentiment, 0)
                category_chunks.setdefault(category, []).append((chunk, counts))

        for category, counts in category_signals.items():
            trend = classify_trend(counts)
            excerpt = get_best_excerpt(category_chunks.get(category, []), category)
            embedding = generate_embedding(excerpt)

            results.append(
                {
                    "category": category,
                    "trend": trend,
                    "reason": _build_reason(category, counts),
                    "source": (
                        f"{document.source_name} - {published_date}" if published_date else document.source_name
                    ),
                    "rag_excerpt": excerpt,
                    "source_link": document.source_link,
                    "embedding_vector": embedding,
                    "timestamp": published_date,
                }
            )

    return results


__all__ = [
    "analyze_retail_categories",
    "chunk_text",
    "classify_trend",
    "extract_main_text",
    "extract_published_date",
    "fetch_url",
    "generate_embedding",
    "get_best_excerpt",
    "score_signals",
]
