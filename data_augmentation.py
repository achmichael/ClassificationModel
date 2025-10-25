"""Lightweight text augmentation utilities for the halal/haram classifier.

The augmentation routines introduce minor lexical variation without altering
labels. They are intentionally conservative to avoid semantic drift that could
confuse the downstream model.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd
from nltk.corpus import wordnet

from text_preprocessing import ensure_nltk_resources, preprocess_text

ensure_nltk_resources(("wordnet", "omw-1.4"))

NEUTRAL_WORDS = {
    "and",
    "with",
    "contains",
    "containing",
    "including",
    "made",
    "from",
    "of",
    "the",
    "a",
    "an",
}

PROTECTED_WORDS = {
    "halal",
    "haram",
    "babi",
    "pork",
    "pig",
    "gelatin",
    "gelatine",
    "alcohol",
    "wine",
    "beer",
    "soy",
    "soybean",
    "lecithin",
    "soy lecithin",
}

MAX_AUG_PER_SAMPLE = 3


def _get_wordnet_synonyms(word: str) -> List[str]:
    """Collect WordNet synonyms for the given word."""
    synonyms: set[str] = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ").lower()
            if candidate != word:
                synonyms.add(candidate)
    return list(synonyms)


def _random_synonym_replacement(text: str, rng: random.Random) -> str:
    words = text.split()
    candidates = [idx for idx, token in enumerate(words) if token.isalpha() and token not in PROTECTED_WORDS]
    rng.shuffle(candidates)
    for idx in candidates:
        synonyms = _get_wordnet_synonyms(words[idx])
        filtered = [syn for syn in synonyms if syn not in PROTECTED_WORDS]
        if not filtered:
            continue
        words[idx] = rng.choice(filtered)
        return " ".join(words)
    return text


def _random_swap(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    idx1, idx2 = rng.sample(range(len(words)), 2)
    words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)


def _random_deletion(text: str, rng: random.Random, drop_probability: float = 0.3) -> str:
    words = text.split()
    if len(words) <= 2:
        return text
    kept: List[str] = []
    for token in words:
        if token in NEUTRAL_WORDS and rng.random() < drop_probability:
            continue
        kept.append(token)
    if not kept:
        kept = words[:1]
    return " ".join(kept)


AugmentFunc = Callable[[str, random.Random], str]

AUGMENTATION_PIPELINE: Tuple[Tuple[str, AugmentFunc], ...] = (
    ("synonym_replacement", _random_synonym_replacement),
    ("random_swap", _random_swap),
    ("random_deletion", _random_deletion),
)


def augment_dataset(
    df: pd.DataFrame,
    random_state: int = 42,
    max_aug_per_sample: int = MAX_AUG_PER_SAMPLE,
) -> pd.DataFrame:
    """Generate augmented samples for each row in the provided DataFrame."""
    rng = random.Random(random_state)
    augmented_rows: List[Dict[str, object]] = []

    for idx, row in df.iterrows():
        original_text = str(row.get("text", "") or row.get("clean_text", ""))
        if not original_text:
            continue

        generated = 0
        for aug_name, aug_func in AUGMENTATION_PIPELINE:
            if generated >= max_aug_per_sample:
                break
            augmented_text = aug_func(original_text, rng)
            if not augmented_text or augmented_text.lower() == original_text.lower():
                continue
            clean_text = preprocess_text(augmented_text)
            if not clean_text or clean_text == row.get("clean_text", ""):
                continue

            record = row.to_dict()
            record["text"] = augmented_text
            record["clean_text"] = clean_text
            record["augmentation_type"] = aug_name
            record["is_augmented"] = True
            record["source_index"] = idx
            augmented_rows.append(record)
            generated += 1

    if not augmented_rows:
        return pd.DataFrame(columns=list(df.columns) + ["augmentation_type", "is_augmented", "source_index"])

    return pd.DataFrame(augmented_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment ingredient texts with lightweight perturbations.")
    parser.add_argument("--input", type=Path, default=Path("data/preprocessed_dataset.csv"), help="Path to the base dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/augmented_dataset.csv"),
        help="Destination path for the combined augmented dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--max-aug-per-sample",
        type=int,
        default=MAX_AUG_PER_SAMPLE,
        help="Max augmented variants generated per original row",
    )
    args = parser.parse_args()

    print(f"Loading dataset from {args.input}...")
    base_df = pd.read_csv(args.input)
    print(f"✓ Loaded {len(base_df)} rows")

    augmented_df = augment_dataset(base_df, random_state=args.seed, max_aug_per_sample=args.max_aug_per_sample)
    print(f"Generated {len(augmented_df)} augmented rows")

    augmented_df = augmented_df.assign(is_augmented=True)
    base_augmented_flags = {
        "augmentation_type": "original",
        "is_augmented": False,
        "source_index": base_df.index,
    }
    base_with_flags = base_df.assign(**base_augmented_flags)

    combined = pd.concat([base_with_flags, augmented_df], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)

    print(f"\n✓ Combined dataset saved to: {args.output}")
    print(f"  Original rows : {len(base_df)}")
    print(f"  Augmented rows: {len(augmented_df)}")
    print(f"  Total rows    : {len(combined)}")


if __name__ == "__main__":
    main()
