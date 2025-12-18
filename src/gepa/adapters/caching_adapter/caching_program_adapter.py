from typing import Any, Callable, TypedDict

import logging
import os
import shutil
import tempfile
from dotenv import load_dotenv
from gepa import EvaluationBatch, GEPAAdapter
from gepa.adapters.caching_adapter.openevolve_evaluator import evaluate as openevolve_evaluate

FAILED_SCORE = 0.0


logger = logging.getLogger(__name__)

class CachingAdapter(GEPAAdapter[Any, Any, Any]):
    """Minimal adapter that wires OpenEvolve evaluator into GEPA."""
    def __init__(
        self,
        model: str | Callable,
        failure_score: float = FAILED_SCORE,
        max_litellm_workers: int = 1,
    ):
        if isinstance(model, str):
            import litellm  # type: ignore

            self.litellm = litellm
            model_name = model

            def _call_lm(prompt: str) -> str:
                load_dotenv()
                logger.info(f"LM Prompt: {prompt}")
                completion = self.litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                logger.info(f"LM Response: {completion.choices[0].message.content}")
                return completion.choices[0].message.content or ""

            self.reflection_lm = _call_lm
        else:
            self.reflection_lm = model
        self.failure_score = failure_score
        self.max_litellm_workers = max_litellm_workers
        self._last_tmpdir: str | None = None

    def evaluate(
            self, 
            batch: list[dict], # Each item: {"trace_path": str, "trace_id": int} (a subset of the complete dataset)
            candidate: dict[str, str], # {"program": code}
            capture_traces: bool = False
    ) -> EvaluationBatch[Any, Any]:
        """
        Evaluate on a minibatch of samples.
        
        IMPORTANT: trajectories returned are never used (see api.py)
        """
        # success_rolloout_outputs ({"trace_id": id, "hit_rate": float}) 
        # or failure_rollout_outputs ({"trace_id": trace_id, "error": str})
        outputs: list[dict] = [] 
        # a list of hit rates (or failure_score)
        scores: list[float] = []
        # {"runs_successfully": float, "trace_id": id, "hit_rate": float | "error": str}
        trajectories: list[dict] | None = [] if capture_traces else None

        code = candidate["program"]

        for item in batch:
            trace_path = item["trace_path"]
            try: 
                single_result_dict = openevolve_evaluate(code, trace_path)
            except Exception as e:
                single_result_dict = {'runs_successfully': 0.0, 'error': str(e)}
            if single_result_dict['runs_successfully'] == 0.0:
                single_output = {"trace_id": item["trace_id"], "error": single_result_dict['error']}
                single_score = self.failure_score
                single_traj = {"trace_id": item["trace_id"], "runs_successfully": False, "error": single_result_dict['error']}
            else:
                single_score = single_result_dict[os.path.basename(item['trace_path'])]
                single_output = {"trace_id": item["trace_id"], "hit_rate": single_score}
                single_traj = {"trace_id": item["trace_id"], "runs_successfully": True, "hit_rate": single_score}
            outputs.append(single_output)
            scores.append(single_score)
            if capture_traces:
                trajectories.append(single_traj)


        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )
    
    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ):
        """Return the aggregated feedback dataset for the "program" component."""
        ret_d: dict[str, list[dict[str, Any]]] = {} # component_name, items
        
        assert len(components_to_update) == 1
        comp = components_to_update[0]
        
        items: list[dict[str, Any]] = [] # dataset_with_feedback for "program": a list of samples' feedbacks (feedback = {...})
        
        trace_instances = list(zip(eval_batch.trajectories, eval_batch.scores, eval_batch.outputs, strict=False))

        for trace_instance in trace_instances:
            traj, _, _ = trace_instance
            runs_successfully = traj["runs_successfully"]
            if runs_successfully:
                d = {
                    "Trace ID": traj["trace_id"],
                    "Hit Rate": round(traj["hit_rate"], 4),
                }
            else:
                d = {
                    "Trace ID": traj["trace_id"],
                    "Error": traj["error"],
                }
            items.append(d)

        ret_d[comp] = items
        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d


    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        Use reflection LM and feedback dataset to rewrite the candidate "program" 
        
        Prompt already has the domain-specific instructions (Strategy API, SPOT/ON_DEMAND, etc.)
        """
        from gepa.adapters.caching_adapter.caching_proposal_signature import (
            OpenEvolveProposalSignature,
        )

        new_texts: dict[str, str] = {}
        for name in components_to_update:
            base_instruction = candidate[name] # the code_str
            dataset_with_feedback = reflective_dataset.get(name, []) # the dataset gen by make_reflective_dataset

            new_texts[name] = OpenEvolveProposalSignature.run(
                lm=self.reflection_lm,
                input_dict={
                    "current_instruction_doc": base_instruction,
                    "dataset_with_feedback": dataset_with_feedback,
                },
            )["new_program"]

        return new_texts
