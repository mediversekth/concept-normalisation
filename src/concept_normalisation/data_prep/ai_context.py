"""
Generate AI-written semantic context using Ollama.
"""

from pathlib import Path
from typing import Any
import time

import ollama
import pandas as pd


DEFAULT_SYSTEM_PROMPT = """
You create concise factual clinical descriptions for terminology
mapping to SNOMED CT.

You receive:
- the source table name;
- the query column name;
- the query value;
- the selected context column names;
- the values found in those context columns.

Write one or two clear sentences describing the query value using only
the supplied information.

Do not infer missing facts.
Do not invent diagnoses.
Do not add information that was not provided.
Return only the description without preamble.
""".strip()


def build_ai_prompt(
    metadata: dict[str, Any],
) -> str:
    """
    Convert structured metadata into an Ollama prompt.
    """

    table_name = (metadata.get("table_name") or "unknown table")

    query_column = metadata["query_column"]

    query_value = metadata.get(
        "query_value",
        "",
    )

    context_columns = metadata.get(
        "context_columns",
        [],
    )

    context_values = metadata.get(
        "context_values",
    ) or {}

    lines = [
        f"Table name: {table_name}",
        f"Query column: {query_column}",
        f"Query value: {query_value or 'missing'}",
        (
            "Context columns: "
            + (
                ", ".join(context_columns)
                if len(context_columns) > 0
                else "none"
            )
        ),
        "Context values:",
    ]

    if len(context_columns) == 0:
        lines.append("- none")

    for column in context_columns:
        value = context_values.get(
            column,
            "",
        )

        lines.append(
            f"- {column}: {value or 'missing'}"
        )

    return "\n".join(lines)


# Two hand-written (query value, ordered context values, ideal description)
# examples, in the *same* "column names + values" style build_ai_prompt
# already produces -- this is "prompt 1" / "named" from the earlier
# prompt-type x shot-type x model experiment, which came out on top for
# both internal consistency and mean similarity. Stored generically (no
# column names baked in) so the same two examples work regardless of
# which table/query_column/context_columns this is pointed at -- see
# _build_example_metadata below for how they get mapped onto whatever the
# current row's metadata actually uses.
FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "query_value": "URINE CULTURE",
        "context_values": ["URINE", "ESCHERICHIA COLI", "AMPICILLIN"],
        "ideal_description": (
            "A urine culture test that grew Escherichia coli, "
            "subsequently tested for susceptibility to ampicillin."
        ),
    },
    {
        "query_value": "BLOOD CULTURE, ROUTINE",
        "context_values": ["BLOOD CULTURE", "STAPHYLOCOCCUS AUREUS", "VANCOMYCIN"],
        "ideal_description": (
            "A routine blood culture test that grew Staphylococcus "
            "aureus, subsequently tested for susceptibility to vancomycin."
        ),
    },
]


def _build_example_metadata(
    example: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Maps a generic (query_value, context_values) example onto whatever
    table_name/query_column/context_columns the CURRENT row's metadata
    uses, so the few-shot example always matches the real prompt's
    structure -- reusable across any table without editing this file."""

    context_columns = metadata.get("context_columns", [])
    context_values = list(example["context_values"])

    padded = context_values + [""] * max(0, len(context_columns) - len(context_values))
    padded = padded[: len(context_columns)]

    return {
        "table_name": metadata.get("table_name"),
        "query_column": metadata.get("query_column"),
        "query_value": example["query_value"],
        "context_columns": context_columns,
        "context_values": dict(zip(context_columns, padded)),
    }


def build_few_shot_messages(metadata: dict[str, Any]) -> list[dict[str, str]]:
    """Turns FEW_SHOT_EXAMPLES into chat turns, formatted to match the
    current row's actual query_column/context_columns."""

    messages: list[dict[str, str]] = []

    for example in FEW_SHOT_EXAMPLES:
        example_metadata = _build_example_metadata(example, metadata)
        example_prompt = build_ai_prompt(example_metadata)

        messages.append({"role": "user", "content": example_prompt})
        messages.append({"role": "assistant", "content": example["ideal_description"]})

    return messages


def generate_ai_context(
    metadata: dict[str, Any],
    model: str = "llama3.1",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    few_shot: bool = True,
) -> str:
    """
    Generate one AI context description.

    few_shot=True (the default) uses the two FEW_SHOT_EXAMPLES above --
    this is the prompt_type="named" + shot_type="few" combination the
    earlier experiment found gave the most internally-consistent results
    with the highest similarity to its top SNOMED match. Pass
    few_shot=False to go back to a plain zero-shot call.
    """

    user_prompt = build_ai_prompt(metadata)

    messages = [{"role": "system", "content": system_prompt}]

    if few_shot:
        messages.extend(build_few_shot_messages(metadata))

    messages.append({"role": "user", "content": user_prompt})

    response = ollama.chat(
        model=model,
        messages=messages,
    )

    return response["message"]["content"].strip()


def metadata_key(metadata: dict[str, Any]) -> str:
    """
    A stable string key identifying what a metadata dict would produce --
    used both to dedupe identical AI calls (the same query/context
    combination appearing on many rows -- common on tables like
    diagnosis_icd9_snomed where the same diagnosisstring recurs across
    many patients) and to checkpoint/resume progress.
    """

    context_values = metadata.get("context_values") or {}

    key_parts = [
        str(metadata.get("table_name", "")),
        str(metadata.get("query_column", "")),
        str(metadata.get("query_value", "")),
    ]

    for column in sorted(context_values):
        key_parts.append(f"{column}={context_values[column]}")

    return "|".join(key_parts)


def generate_ai_contexts(
    metadata_items,
    model: str = "llama3.1",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    few_shot: bool = True,
    show_progress: bool = True,
    progress_every: int = 50,
    checkpoint_path: "str | Path | None" = None,
    max_attempts: int = 3,
) -> list[str]:
    """
    Generate AI context for multiple metadata dictionaries.

    Pass checkpoint_path to append each result to a CSV (keyed by
    metadata_key) as it's produced, AND to resume from it: if the file
    already exists, entries already present are loaded and skipped
    instead of being regenerated -- so an interrupted run picks back up
    where it left off rather than starting over.

    A transient failure (e.g. a model-load timeout) retries with backoff
    up to max_attempts times before giving that one item "[ERROR] ..." as
    its result, rather than aborting the whole run.
    """

    metadata_items = list(metadata_items)
    total = len(metadata_items)

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    done: dict[str, str] = {}

    if checkpoint_path is not None and checkpoint_path.exists():
        existing = pd.read_csv(checkpoint_path)
        done = dict(zip(existing["key"], existing["ai_output"]))
        print(
            f"Resuming from checkpoint: {len(done):,} already done, "
            f"{total - len(done):,} remaining"
        )

    results = []

    for position, metadata in enumerate(metadata_items, start=1):
        key = metadata_key(metadata)

        if key in done:
            results.append(done[key])
            continue

        result = None
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                result = generate_ai_context(
                    metadata=metadata,
                    model=model,
                    system_prompt=system_prompt,
                    few_shot=few_shot,
                )
                break
            except Exception as error:
                last_error = error
                if attempt < max_attempts:
                    wait_seconds = 5 * attempt
                    print(
                        f"  call failed ({error}), retrying in {wait_seconds}s "
                        f"(attempt {attempt}/{max_attempts})..."
                    )
                    time.sleep(wait_seconds)

        if result is None:
            print(f"  giving up after {max_attempts} attempts: {last_error}")
            result = f"[ERROR] {last_error}"

        done[key] = result
        results.append(result)

        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"key": key, "ai_output": result}]).to_csv(
                checkpoint_path,
                mode="a",
                header=not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0,
                index=False,
            )

        if show_progress and (
            position % progress_every == 0 or position == total
        ):
            print(f"AI context: {position:,}/{total:,}")

    return results
