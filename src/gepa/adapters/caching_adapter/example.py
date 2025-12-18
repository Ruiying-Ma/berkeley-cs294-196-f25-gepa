from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Optional

from gepa.adapters.caching_adapter.caching_program_adapter import CachingAdapter
from gepa.utils import MaxIterationsStopper
import os

INITIAL_PROGRAM_SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lru.py")
with open(INITIAL_PROGRAM_SRC_PATH, "r", encoding="utf-8") as f:
    INITIAL_PROGRAM_SRC = f.read().strip()

def load_dataset():
    """Load train/val/test splits from extracted cant-be-late traces."""

    trace_root_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace")
    dataset = []
    for trace_file in sorted(os.listdir(trace_root_folder)):
        item = {
            "trace_id": len(dataset),
            "trace_path": os.path.join(trace_root_folder, trace_file),
        }
        dataset.append(item)
    return dataset, dataset, dataset

def _resolve_run_dir() -> Path:
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gepa_results")
    return Path(run_dir)

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
    # https://docs.litellm.ai/docs/providers/openai#openai-chat-completion-models
    # https://docs.litellm.ai/docs/providers/gemini#gemini-3-models---thinking_level-parameter
    MODEL = "gpt-5"
    # MODEL = "gemini/gemini-3-pro-preview"
    MAX_ITERATIONS = 100


    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    trace_ratio = 1.0

    run_dir = _resolve_run_dir()
    print(f"GEPA artifacts will be saved to: {run_dir}")

    
    adapter = CachingAdapter(
        model=MODEL
    )
    
    train_set, val_set, test_set = load_dataset()
    print(f"Dataset sizes -> train/val: {len(train_set)} samples")
    
    from gepa import optimize
    iteration_stopper = MaxIterationsStopper(max_iterations=MAX_ITERATIONS)
    stop_callbacks = iteration_stopper

    gepa_result = optimize(
        seed_candidate={"program": INITIAL_PROGRAM_SRC},
        trainset=train_set,
        adapter=adapter,
        reflection_lm=MODEL,
        run_dir=str(run_dir),
        reflection_minibatch_size=3,
        stop_callbacks=stop_callbacks,
        skip_perfect_score=False,
    )
    best_candidate = gepa_result.best_candidate
    print(f"Best program from optimization: {best_candidate['program']}")
    best_score = gepa_result.val_aggregate_scores[gepa_result.best_idx]
    print(f"Best program validation score: {best_score}")
    output_optimized = adapter.evaluate(test_set, best_candidate)
    optimized_score = (sum(output_optimized.scores) / max(1, len(output_optimized.scores))) if output_optimized.scores else None
    print(f"Optimized program test score (avg over test): {optimized_score}")
