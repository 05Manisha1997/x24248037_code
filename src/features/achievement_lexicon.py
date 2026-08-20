from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Lexicons derived from achievement motivation & impression management research
COMPETENCE_TERMS = {
    "skilled", "talented", "capable", "competent", "expert", "proficient", "mastered",
    "qualified", "accomplished", "brilliant", "intelligent", "smart", "gifted",
    "hardworking", "dedicated", "disciplined", "focused", "productive", "efficient",
    "high achieving", "high-achieving", "top performer", "best", "excel", "excelled",
    "outstanding", "exceptional", "remarkable", "impressive", "proven", "experienced",
}

SUCCESS_TERMS = {
    "success", "successful", "achieved", "achievement", "accomplish", "accomplished",
    "won", "winner", "victory", "triumph", "milestone", "breakthrough", "promoted",
    "graduated", "certified", "awarded", "honored", "recognized", "celebrated",
    "goal", "goals", "dream", "dreams", "ambition", "ambitious", "career", "promotion",
    "scholarship", "degree", "certification", "launched", "funded", "profitable",
    "record", "milestone", "first place", "champion", "medal", "trophy", "prize",
}

STATUS_TERMS = {
    "luxury", "exclusive", "premium", "elite", "vip", "influencer", "famous",
    "popular", "followers", "verified", "brand", "collaboration", "sponsored",
    "lifestyle", "wealthy", "rich", "millionaire", "ceo", "founder", "entrepreneur",
    "executive", "leader", "leadership", "prestigious", "high-end", "designer",
    "first class", "penthouse", "yacht", "rolex", "gucci", "louis vuitton",
    "blessed", "grateful", "humble brag", "living my best life", "blessed and",
}

IDEALISATION_MARKERS = {
    "perfect", "flawless", "amazing", "incredible", "unbelievable", "best day ever",
    "couldn't be happier", "living the dream", "never felt better", "so blessed",
    "grateful for everything", "#blessed", "picture perfect", "goals af",
}

DISTRESS_TERMS = {
    "depressed", "anxiety", "anxious", "stressed", "overwhelmed", "exhausted",
    "burnout", "lonely", "hopeless", "worthless", "suicidal", "crying", "tired",
    "can't sleep", "insomnia", "panic", "sad", "miserable", "empty", "numb",
}

SOCIAL_COMPARISON = {
    "everyone else", "compared to", "better than", "worse than", "falling behind",
    "not good enough", "they all", "everyone is", "fomo", "keeping up",
}


@dataclass
class AOSPFeatures:
    competence_score: float
    success_score: float
    status_score: float
    idealisation_score: float
    distress_score: float
    social_comparison_score: float
    aosp_composite: float
    paradox_index: float  # high presentation × distress
    word_count: int
    exclamation_density: float
    hashtag_count: int
    emoji_density: float


def _count_lexicon_hits(text: str, terms: set[str]) -> int:
    text_lower = text.lower()
    hits = 0
    for term in terms:
        if " " in term:
            hits += text_lower.count(term)
        else:
            hits += len(re.findall(rf"\b{re.escape(term)}\b", text_lower))
    return hits


def _emoji_count(text: str) -> int:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE,
    )
    return len(emoji_pattern.findall(text))


def extract_aosp_features(text: str) -> AOSPFeatures:
    if not isinstance(text, str) or not text.strip():
        return AOSPFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    words = text.split()
    word_count = max(len(words), 1)

    competence = _count_lexicon_hits(text, COMPETENCE_TERMS) / word_count
    success = _count_lexicon_hits(text, SUCCESS_TERMS) / word_count
    status = _count_lexicon_hits(text, STATUS_TERMS) / word_count
    idealisation = _count_lexicon_hits(text, IDEALISATION_MARKERS) / word_count
    distress = _count_lexicon_hits(text, DISTRESS_TERMS) / word_count
    comparison = _count_lexicon_hits(text, SOCIAL_COMPARISON) / word_count

    aosp_composite = 0.34 * competence + 0.33 * success + 0.33 * status + 0.15 * idealisation
    paradox_index = aosp_composite * distress  # high idealised presentation + distress

    exclamations = text.count("!") / word_count
    hashtags = len(re.findall(r"#\w+", text))
    emoji_density = _emoji_count(text) / word_count

    return AOSPFeatures(
        competence_score=competence,
        success_score=success,
        status_score=status,
        idealisation_score=idealisation,
        distress_score=distress,
        social_comparison_score=comparison,
        aosp_composite=aosp_composite,
        paradox_index=paradox_index,
        word_count=word_count,
        exclamation_density=exclamations,
        hashtag_count=hashtags,
        emoji_density=emoji_density,
    )


def extract_aosp_dataframe(texts: pd.Series) -> pd.DataFrame:
    records = [extract_aosp_features(t).__dict__ for t in texts.fillna("")]
    return pd.DataFrame(records)


AOSP_COLUMN_NAMES = [
    "competence_score", "success_score", "status_score", "idealisation_score",
    "distress_score", "social_comparison_score", "aosp_composite", "paradox_index",
    "word_count", "exclamation_density", "hashtag_count", "emoji_density",
]
