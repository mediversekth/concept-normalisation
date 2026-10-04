"""
Prepare the PhysioNet SNOMED CT Entity Linking Challenge data for the
concept-normalisation pipeline.

Output columns:
    diagnosisstring
    snomed

Every diagnosisstring is enriched with local context taken from the note
around the exact annotation start/end offsets:

    "<span> - <local note context>"

The number of UNIQUE diagnosis strings can optionally be limited before the
slow GraphRAG pipeline is run.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _paragraph_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Return the paragraph containing the annotated character span."""
    left = text.rfind("\n\n", 0, start)
    left = 0 if left == -1 else left + 2

    right = text.find("\n\n", end)
    right = len(text) if right == -1 else right

    return left, right


def _centered_word_window(
    text: str,
    mention_start: int,
    mention_end: int,
    max_words: int,
) -> str:
    """
    Return at most max_words words, centred on the annotated mention.

    mention_start and mention_end are offsets relative to `text`.
    """
    words = list(re.finditer(r"\S+", text))
    if not words:
        return ""

    mention_word_indices = [
        i
        for i, word in enumerate(words)
        if word.end() > mention_start and word.start() < mention_end
    ]

    if not mention_word_indices:
        # Fallback if offsets land strangely around whitespace.
        closest = min(
            range(len(words)),
            key=lambda i: abs(words[i].start() - mention_start),
        )
        mention_word_indices = [closest]

    first = mention_word_indices[0]
    last = mention_word_indices[-1]

    mention_word_count = last - first + 1
    remaining = max(0, max_words - mention_word_count)

    before = remaining // 2
    after = remaining - before

    window_start = max(0, first - before)
    window_end = min(len(words), last + after + 1)

    # Reuse unused budget if one side reached the paragraph boundary.
    while window_end - window_start < min(max_words, len(words)):
        if window_start > 0:
            window_start -= 1
        elif window_end < len(words):
            window_end += 1
        else:
            break

    char_start = words[window_start].start()
    char_end = words[window_end - 1].end()

    return text[char_start:char_end]


def extract_context(
    note_text: str,
    start: int,
    end: int,
    max_words: int = 30,
) -> str:
    """
    Extract a small amount of context around the exact annotation offsets.

    The paragraph containing start:end is used first, then capped to a
    max_words window centred on the annotated mention.
    """
    if not isinstance(note_text, str) or not note_text:
        return ""

    start = max(0, min(int(start), len(note_text)))
    end = max(start, min(int(end), len(note_text)))

    paragraph_start, paragraph_end = _paragraph_bounds(note_text, start, end)
    paragraph = note_text[paragraph_start:paragraph_end]

    local_start = start - paragraph_start
    local_end = end - paragraph_start

    context = _centered_word_window(
        paragraph,
        mention_start=local_start,
        mention_end=local_end,
        max_words=max_words,
    )

    return " ".join(context.split())


def build_diagnosisstring(
    span: str,
    note_text: str,
    start: int,
    end: int,
    *,
    max_context_words: int = 30,
) -> str:
    """
    Build:
        "<span> - <context>"

    Context is always added and is extracted using the exact start/end
    annotation offsets.
    """
    span = " ".join(str(span).split())

    context = extract_context(
        note_text=note_text,
        start=start,
        end=end,
        max_words=max_context_words,
    )

    if not context:
        return span

    return f"{span} - {context}"


def limit_unique_queries(
    data: pd.DataFrame,
    max_unique_queries: int | None,
    *,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Limit the output by unique diagnosisstring values.

    This is useful because GraphRAG is slow and running every unique query
    can take a very long time.
    """
    if max_unique_queries is None:
        return data

    if max_unique_queries <= 0:
        raise ValueError("max_unique_queries must be greater than 0")

    unique_queries = data["diagnosisstring"].drop_duplicates()

    if len(unique_queries) <= max_unique_queries:
        return data

    rng = np.random.default_rng(random_state)
    selected_indices = rng.choice(
        len(unique_queries),
        size=max_unique_queries,
        replace=False,
    )

    selected_queries = set(unique_queries.iloc[selected_indices])

    return data[data["diagnosisstring"].isin(selected_queries)].copy()


def prepare_snomed_challenge(
    notes_path: str | Path,
    annotations_path: str | Path,
    *,
    max_unique_queries: int | None = None,
    random_state: int = 0,
    max_context_words: int = 30,
) -> pd.DataFrame:
    """
    Create the two-column dataset expected by the pipeline:

        diagnosisstring,snomed
    """
    notes = pd.read_csv(notes_path)

    annotations = pd.read_csv(
        annotations_path,
        dtype={"concept_id": str},
    )

    required_note_columns = {"note_id", "text"}
    required_annotation_columns = {
        "note_id",
        "start",
        "end",
        "span",
        "concept_id",
    }

    missing_notes = required_note_columns - set(notes.columns)
    missing_annotations = required_annotation_columns - set(annotations.columns)

    if missing_notes:
        raise ValueError(f"Missing note columns: {sorted(missing_notes)}")

    if missing_annotations:
        raise ValueError(
            f"Missing annotation columns: {sorted(missing_annotations)}"
        )

    data = annotations.merge(
        notes[["note_id", "text"]],
        on="note_id",
        how="left",
        validate="many_to_one",
    )

    if data["text"].isna().any():
        missing_count = data.loc[data["text"].isna(), "note_id"].nunique()
        raise ValueError(
            f"{missing_count} annotated note_id values have no matching note text"
        )

    data["start"] = data["start"].astype(int)
    data["end"] = data["end"].astype(int)

    data["diagnosisstring"] = data.apply(
        lambda row: build_diagnosisstring(
            span=row["span"],
            note_text=row["text"],
            start=row["start"],
            end=row["end"],
            max_context_words=max_context_words,
        ),
        axis=1,
    )

    data["snomed"] = data["concept_id"].astype(str)

    # Only keep the columns needed by the existing pipeline.
    data = data[["diagnosisstring", "snomed"]]

    total_unique = data["diagnosisstring"].nunique()

    data = limit_unique_queries(
        data,
        max_unique_queries=max_unique_queries,
        random_state=random_state,
    )

    final_unique = data["diagnosisstring"].nunique()

    print(f"Rows: {len(data):,}")
    print(f"Unique diagnosis strings: {final_unique:,} / {total_unique:,}")

    if max_unique_queries is not None:
        print(
            f"Unique-query limit: {max_unique_queries:,} "
            f"(random_state={random_state})"
        )

    return data.reset_index(drop=True)


def prepare_snomed_challenge_csv(
    notes_path: str | Path,
    annotations_path: str | Path,
    output_path: str | Path,
    **kwargs,
) -> pd.DataFrame:
    """Prepare and save the pipeline-ready CSV."""
    data = prepare_snomed_challenge(
        notes_path=notes_path,
        annotations_path=annotations_path,
        **kwargs,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SNOMED entity-linking challenge data."
    )

    parser.add_argument("--notes", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    parser.add_argument(
        "--max-unique-queries",
        type=int,
        default=None,
        help=(
            "Maximum number of unique diagnosisstring values to keep. "
            "Useful for slow GraphRAG experiments."
        ),
    )

    parser.add_argument(
        "--max-context-words",
        type=int,
        default=0,
        help="Maximum number of note-context words added to each span.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Seed used when selecting a limited subset of unique queries.",
    )

    args = parser.parse_args()

    prepare_snomed_challenge_csv(
        notes_path=args.notes,
        annotations_path=args.annotations,
        output_path=args.output,
        max_unique_queries=args.max_unique_queries,
        random_state=args.random_state,
        max_context_words=args.max_context_words,
    )


if __name__ == "__main__":
    main()
