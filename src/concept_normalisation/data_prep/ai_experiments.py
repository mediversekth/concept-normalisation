"""
Experiment harness for the AI-context step (algorithm_ai).

Varies three things independently and records every combination so they
can be compared side by side:

1. Prompt type:
   - "named"       -- query column name + value, and context column
                       names + values (this is the current production
                       prompt, see ai_context.build_ai_prompt).
   - "values_only" -- the same values, but with no column names/labels
                       at all.
   - "full_row"    -- every column present in the source row (not just
                       the selected query/context columns), with names.

2. Shot type:
   - "zero" -- just the system prompt + the row's prompt.
   - "few"  -- two hand-written (prompt, ideal description) examples in
              the *same* prompt style are inserted as prior chat turns
              before the row's prompt.

3. Model: any Ollama model name you have pulled locally.

Usage (see the matching notebook cell):

    from src.concept_normalisation.data_prep.ai_experiments import run_ai_experiment

    results = run_ai_experiment(
        data=data,
        query_column=query_column,
        context_columns=context_columns,
        models=["llama3.1", "mistral"],
        n_samples=8,
    )
"""

from pathlib import Path
from typing import Any

import ollama
import pandas as pd

from src.concept_normalisation.data_prep.context_builder import clean_value
from src.concept_normalisation.data_prep.ai_context import DEFAULT_SYSTEM_PROMPT


# Columns the pipeline itself adds on top of the source table. These are
# excluded from the "full_row" prompt type -- that variant is meant to
# show the model the *source* data, not the pipeline's own derived
# columns (which would be circular / noise for this purpose).
DERIVED_COLUMNS = {
    "algorithm_1_text",
    "algorithm_2_text",
    "algorithm_ai_text",
    "algorithm_ai_metadata",
    "algorithm_1_matches",
    "algorithm_2_matches",
    "algorithm_ai_matches",
    "multi_match_matches",
    "elastic_fuzzy_matches",
    "jaccard_matches",
}


# Two small hand-written rows (and their ideal descriptions) used to
# build the few-shot examples. Written for the microbiologyevents
# table -- edit these if you point the experiment at a different table.
FEW_SHOT_EXAMPLE_ROWS: list[dict[str, Any]] = [
    {
        "test_name": "URINE CULTURE",
        "spec_type_desc": "URINE",
        "org_name": "ESCHERICHIA COLI",
        "ab_name": "AMPICILLIN",
        "ideal_description": (
            "A urine culture test that grew Escherichia coli, "
            "subsequently tested for susceptibility to ampicillin."
        ),
    },
    {
        "test_name": "BLOOD CULTURE, ROUTINE",
        "spec_type_desc": "BLOOD CULTURE",
        "org_name": "STAPHYLOCOCCUS AUREUS",
        "ab_name": "VANCOMYCIN",
        "ideal_description": (
            "A routine blood culture test that grew Staphylococcus "
            "aureus, subsequently tested for susceptibility to "
            "vancomycin."
        ),
    },
]


# ------------------------------------------------------------
# Prompt builders -- one per prompt type
# ------------------------------------------------------------

def build_prompt_named(
    row: pd.Series,
    query_column: str,
    context_columns: list[str],
) -> str:
    """Column names + values. Same shape as the production prompt."""

    query_value = clean_value(row[query_column])

    lines = [
        f"Query column: {query_column}",
        f"Query value: {query_value or 'missing'}",
        "Context values:",
    ]

    if not context_columns:
        lines.append("- none")

    for column in context_columns:
        value = clean_value(row[column])
        lines.append(f"- {column}: {value or 'missing'}")

    return "\n".join(lines)


def build_prompt_values_only(
    row: pd.Series,
    query_column: str,
    context_columns: list[str],
) -> str:
    """Just the raw values, no column names/labels at all."""

    query_value = clean_value(row[query_column])
    values = [query_value] + [
        clean_value(row[column]) for column in context_columns
    ]
    values = [value for value in values if value]

    return "\n".join(values) if values else "missing"


def build_prompt_full_row(
    row: pd.Series,
    query_column: str,
    context_columns: list[str],
    include_names: bool = True,
) -> str:
    """Every column in the source row (not just query/context), with
    names by default. Set include_names=False to test the same thing
    without labels."""

    columns = [
        column for column in row.index if column not in DERIVED_COLUMNS
    ]

    lines = []
    for column in columns:
        value = clean_value(row[column])
        if not value:
            continue
        lines.append(f"{column}: {value}" if include_names else value)

    return "\n".join(lines) if lines else "missing"


PROMPT_BUILDERS = {
    "named": build_prompt_named,
    "values_only": build_prompt_values_only,
    "full_row": build_prompt_full_row,
}


def build_prompt(
    row: pd.Series,
    prompt_type: str,
    query_column: str,
    context_columns: list[str],
) -> str:
    if prompt_type not in PROMPT_BUILDERS:
        raise ValueError(
            f"Unknown prompt_type {prompt_type!r}. "
            f"Choose from {sorted(PROMPT_BUILDERS)}"
        )

    return PROMPT_BUILDERS[prompt_type](
        row=row,
        query_column=query_column,
        context_columns=context_columns,
    )


# ------------------------------------------------------------
# Few-shot examples, built with the same prompt-type functions
# ------------------------------------------------------------

def build_few_shot_messages(
    prompt_type: str,
    query_column: str,
    context_columns: list[str],
) -> list[dict[str, str]]:
    """Turn FEW_SHOT_EXAMPLE_ROWS into chat turns, formatted with
    whichever prompt-type function is being tested, so the example
    style always matches the real prompt style."""

    messages: list[dict[str, str]] = []

    for example in FEW_SHOT_EXAMPLE_ROWS:
        example_row = pd.Series(example)

        example_prompt = build_prompt(
            row=example_row,
            prompt_type=prompt_type,
            query_column=query_column,
            context_columns=[
                column for column in context_columns if column in example_row
            ],
        )

        messages.append({"role": "user", "content": example_prompt})
        messages.append(
            {"role": "assistant", "content": example["ideal_description"]}
        )

    return messages


# ------------------------------------------------------------
# Running one combination / the full grid
# ------------------------------------------------------------

def _chat_with_retries(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_attempts: int = 3,
) -> str:
    """
    Calls ollama.chat, retrying on transient server errors (e.g. "timed
    out waiting for llama-server to start", which happens when Ollama
    is under memory pressure -- often from swapping models too often,
    see run_ai_experiment's loop order). Returns the error message
    (prefixed "[ERROR] ") instead of raising after the last attempt, so
    one bad call doesn't lose every result already generated in a long
    grid run.
    """

    import time

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature},
            )
            return response["message"]["content"].strip()
        except Exception as error:  # ollama.ResponseError, ConnectionError, etc.
            last_error = error
            if attempt < max_attempts:
                wait_seconds = 5 * attempt
                print(
                    f"  [{model}] call failed ({error}), "
                    f"retrying in {wait_seconds}s (attempt {attempt}/{max_attempts})..."
                )
                time.sleep(wait_seconds)

    print(f"  [{model}] giving up after {max_attempts} attempts: {last_error}")
    return f"[ERROR] {last_error}"


def generate_one(
    row: pd.Series,
    prompt_type: str,
    shot_type: str,
    model: str,
    query_column: str,
    context_columns: list[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0,
) -> tuple[str, str]:
    """Returns (prompt_text, ai_output) for one row/prompt_type/shot_type/model."""

    if shot_type not in ("zero", "few"):
        raise ValueError("shot_type must be 'zero' or 'few'")

    prompt = build_prompt(
        row=row,
        prompt_type=prompt_type,
        query_column=query_column,
        context_columns=context_columns,
    )

    messages = [{"role": "system", "content": system_prompt}]

    if shot_type == "few":
        messages.extend(
            build_few_shot_messages(
                prompt_type=prompt_type,
                query_column=query_column,
                context_columns=context_columns,
            )
        )

    messages.append({"role": "user", "content": prompt})

    response = _chat_with_retries(model=model, messages=messages, temperature=temperature)

    return prompt, response


def run_ai_experiment(
    data: pd.DataFrame,
    query_column: str,
    context_columns: list[str],
    models: list[str],
    prompt_types: list[str] = ("named", "values_only", "full_row"),
    shot_types: list[str] = ("zero", "few"),
    n_samples: int | None = 5,
    random_state: int = 0,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0,
    show_progress: bool = True,
    checkpoint_path: "str | Path | None" = None,
) -> pd.DataFrame:
    """
    Runs every (sampled row) x (prompt_type) x (shot_type) x (model)
    combination and returns one long-format DataFrame -- one row per
    combination, with the exact prompt text and the AI output, so
    results can be filtered/pivoted/exported for documentation.

    Pass n_samples=None to use `data` exactly as given (e.g. a sample
    you already curated yourself with a fixed random_state) instead of
    sampling again here.

    The loop is ordered model -> row -> prompt_type -> shot_type
    (model outermost) on purpose: Ollama loads/unloads a model's
    weights each time the requested model changes, so looping models
    innermost would reload weights on almost every single call. With
    model outermost, each model is loaded once and does all of its
    work before the next model is even requested.

    Pass checkpoint_path to append each result to a CSV as it's
    produced, so a crash/interrupt partway through a long run doesn't
    lose the results already generated -- reload with
    pd.read_csv(checkpoint_path) to pick up from what completed.
    """

    if n_samples is None:
        sample = data.reset_index(drop=True)
    else:
        sample = data.sample(n_samples, random_state=random_state).reset_index(drop=True)

    total = len(sample) * len(prompt_types) * len(shot_types) * len(models)
    done = 0

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    records = []

    for model in models:
        for sample_index, row in sample.iterrows():
            query_value = clean_value(row[query_column])

            for prompt_type in prompt_types:
                for shot_type in shot_types:
                    prompt_text, ai_output = generate_one(
                        row=row,
                        prompt_type=prompt_type,
                        shot_type=shot_type,
                        model=model,
                        query_column=query_column,
                        context_columns=context_columns,
                        system_prompt=system_prompt,
                        temperature=temperature,
                    )

                    record = {
                        "sample_index": sample_index,
                        "query_value": query_value,
                        "prompt_type": prompt_type,
                        "shot_type": shot_type,
                        "model": model,
                        "prompt": prompt_text,
                        "ai_output": ai_output,
                    }
                    records.append(record)

                    if checkpoint_path is not None:
                        pd.DataFrame([record]).to_csv(
                            checkpoint_path,
                            mode="a",
                            header=not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0,
                            index=False,
                        )


                    done += 1
                    if show_progress:
                        print(f"AI experiment: {done:,}/{total:,}")

    return pd.DataFrame(records)


def build_experiment_manifest(
    query_column: str,
    context_columns: list[str],
    models: list[str],
    prompt_types: list[str],
    shot_types: list[str],
    n_samples: int | None,
    random_state: int,
    system_prompt: str,
    temperature: float,
) -> dict[str, Any]:
    """
    A small record of exactly what a run's parameters were, meant to be
    saved alongside the results (see the notebook's save cell) so a
    given results file is reproducible/traceable later.
    """

    import datetime

    return {
        "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "query_column": query_column,
        "context_columns": list(context_columns),
        "models": list(models),
        "prompt_types": list(prompt_types),
        "shot_types": list(shot_types),
        "n_samples": n_samples,
        "random_state": random_state,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "few_shot_example_rows": FEW_SHOT_EXAMPLE_ROWS,
    }