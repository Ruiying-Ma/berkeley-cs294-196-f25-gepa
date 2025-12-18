from pathlib import Path
from datetime import datetime
import json
from typing import Optional

from gepa.adapters.multiagent_system_adapter.open_evolve_program_adapter import MultiagentSystemAdapter

INITIAL_PROGRAM_SRC = open(Path(__file__).resolve().parent / "openevolve" / "initial_program.py", "r", encoding="utf-8").read()

# Dataset root should match evaluator's ProgramDevDataset default.
# Evaluator looks for <adapter_root>/programdev; if missing we fallback to example_mas/programdev.
ADAPTER_ROOT = Path(__file__).resolve().parent
PRIMARY_DATASET_ROOT = ADAPTER_ROOT / "programdev"
FALLBACK_DATASET_ROOT = ADAPTER_ROOT / "example_mas" / "programdev"

# Ratio applied to number of tasks sampled for train/val/test
DEFAULT_TRACE_RATIO = 0.30


def load_dataset(
        max_traces_per_split: int | None = None,
        trace_ratio: float | None = None,
        include_test: bool = True,
):
    """Load train/val/test splits from programdev tasks.

    This replaces the previous cant-be-late trace loading. It mirrors the
    evaluator's ProgramDevDataset which consumes ``programdev`` task files.

    Splitting strategy:
    - Enumerate all name/description pairs under the dataset root.
    - Apply ``trace_ratio`` to limit how many tasks are used (minimum DEFAULT_TRACE_RATIO).
    - Apply ``max_traces_per_split`` as an absolute cap.
    - Train and val receive the same sampled set (consistent with prior trace loader design).
    - Test receives either the same sampled set (if ratio < 1.0 or max limit applied) or all tasks.
    - If ``include_test`` is False, test set is empty.

    Each task dict provides keys: ``task_index``, ``task_name``, ``description``.
    """
    # Resolve dataset root similarly to evaluator
    if PRIMARY_DATASET_ROOT.exists():
        dataset_root = PRIMARY_DATASET_ROOT
    elif FALLBACK_DATASET_ROOT.exists():
        dataset_root = FALLBACK_DATASET_ROOT
    else:
        raise FileNotFoundError(
            f"Neither programdev dataset directory found: '{PRIMARY_DATASET_ROOT}' nor fallback '{FALLBACK_DATASET_ROOT}'"
        )

    # Gather tasks (names_*.txt + descriptions_*.txt)
    name_files = sorted(dataset_root.glob("names_*.txt"))
    tasks: list[dict[str, str]] = []
    for name_file in name_files:
        index = name_file.stem.split("_")[1]
        desc_file = dataset_root / f"descriptions_{index}.txt"
        if not desc_file.exists():
            continue
        name = name_file.read_text(encoding="utf-8").strip()
        description = desc_file.read_text(encoding="utf-8").strip()
        if name and description:
            tasks.append({
                "task_index": index,
                "task_name": name,
                "description": description,
            })

    if not tasks:
        raise FileNotFoundError(f"No programdev tasks found under {dataset_root}")

    # Apply ratio and cap
    ratio = DEFAULT_TRACE_RATIO if trace_ratio is None else trace_ratio
    ratio = min(1.0, max(ratio, DEFAULT_TRACE_RATIO))
    desired_count = max(1, round(len(tasks) * ratio))
    sampled = tasks[:desired_count]

    if max_traces_per_split is not None:
        sampled = sampled[:max_traces_per_split]

    # Determine test set
    if (max_traces_per_split is not None) or (ratio < 1.0):
        test_tasks = sampled
    else:
        test_tasks = tasks

    train_set = sampled
    val_set = sampled  # Mirror previous behavior: identical train/val splits
    test_set = test_tasks if include_test else []
    return train_set, val_set, test_set


def _resolve_run_dir() -> Path:
    import os

    run_dir_env = os.environ.get("GEPA_RUN_DIR")
    if run_dir_env:
        run_dir = Path(run_dir_env)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs") / "multiagent_system" / timestamp

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_checkpoints(
        run_dir: Path,
        gepa_result,
        base_score: Optional[float],
        optimized_score: Optional[float],
        best_candidate: dict[str, str],
):
    # Serialize the full GEPA result for later inspection
    result_path = run_dir / "gepa_result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(gepa_result.to_dict(), f, indent=2)

    # Write the best program as a Python file
    best_program_path = run_dir / "best_program.py"
    best_program_path.write_text(best_candidate["program"], encoding="utf-8")

    # Record test metrics for quick reference
    metrics_path = run_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_test_score": base_score,
                "optimized_test_score": optimized_score,
                "best_candidate_index": gepa_result.best_idx,
            },
            f,
            indent=2,
        )

    # Snapshot every candidate for manual analysis
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(exist_ok=True)
    for idx, candidate in enumerate(gepa_result.candidates):
        program_path = candidates_dir / f"candidate_{idx:03d}.py"
        program_path.write_text(candidate["program"], encoding="utf-8")


if __name__ == "__main__":
    import os

    max_traces_env = os.environ.get("MULTAGENT_SYSTEM_MAX_TRACES")
    max_traces = int(max_traces_env) if max_traces_env else None
    max_metric_calls_env = os.environ.get("GEPA_MAX_METRIC_CALLS")
    max_metric_calls = int(max_metric_calls_env) if max_metric_calls_env else 20
    trace_ratio_env = os.environ.get("MULTAGENT_SYSTEM_TRACE_RATIO")
    trace_ratio = DEFAULT_TRACE_RATIO
    if trace_ratio_env:
        try:
            trace_ratio = float(trace_ratio_env)
        except ValueError:
            print(
                f"Invalid MULTAGENT_SYSTEM_TRACE_RATIO='{trace_ratio_env}', falling back to {DEFAULT_TRACE_RATIO:.2f}",
                flush=True,
            )
    trace_ratio = min(1.0, max(trace_ratio, DEFAULT_TRACE_RATIO))
    skip_test = os.environ.get("GEPA_SKIP_TEST", "0") == "1"

    run_dir = _resolve_run_dir()
    print(f"GEPA artifacts will be saved to: {run_dir}")
    print(f"Using {trace_ratio:.0%} of programdev tasks for train/val evaluation", flush=True)

    adapter = MultiagentSystemAdapter(
        model="openai/gpt-4.1-mini", # TODO: do we need a task_lm or not
        # model="openai/gpt-4.1-mini", # this is used for reflection LM
        # model="openai/o3"
    )

    # Load from train and test set
    train_set, val_set, test_set = load_dataset(
        max_traces_per_split=max_traces,
        trace_ratio=trace_ratio,
        include_test=not skip_test,
    )

    if skip_test:
        base_score: Optional[float] = None
        print("Base program score: skipped (GEPA_SKIP_TEST=1)")
    else:
        output_base = adapter.evaluate(test_set, {"program": INITIAL_PROGRAM_SRC})
        base_score = sum(output_base.scores)
        print(f"Base program score: {base_score}")

    # NOTE(core): GEPA optimization
    from gepa import optimize

    gepa_result = optimize(
        seed_candidate={"program": INITIAL_PROGRAM_SRC},
        trainset=train_set,
        valset=val_set,
        adapter=adapter,
        reflection_lm="openai/o3",
        max_metric_calls=max_metric_calls,
        run_dir=str(run_dir),
        reflection_minibatch_size=3,
    )
    best_candidate = gepa_result.best_candidate
    print(f"Best program from optimization: {best_candidate['program']}")

    if skip_test:
        optimized_score: Optional[float] = None
        print("Optimized program score: skipped (GEPA_SKIP_TEST=1)")
    else:
        output_optimized = adapter.evaluate(test_set, best_candidate)
        optimized_score = sum(output_optimized.scores)
        print(f"Optimized program score: {optimized_score}")

    _write_checkpoints(run_dir, gepa_result, base_score, optimized_score, best_candidate)
    print(f"Checkpoint artifacts written under {run_dir}")
