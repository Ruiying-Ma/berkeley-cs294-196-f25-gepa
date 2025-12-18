# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from gepa.proposer.reflective_mutation.base import Signature
from typing import Optional
import re
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)

class OpenEvolveProposalSignature(Signature):
    # Adapt from full_rewrite_prompt_template in openEvolve
    prompt_template = """You are optimizing a cache eviction algorithm to minimize cache miss rates.
    
## Context:
The cache receives a sequence of access requests for objects, and when the cache is full, it must evict an object to make space for a new one. The cache is full when the total number of cached objects reaches its capacity. Focus on improving the `evict` function, the `update_after_hit` function, the `update_after_insert` function, and the `update_after_evict` function to find a cache eviction algorithm with as low miss rate as possible.

## Current Cache Eviction Algorithm Implementation:
```python
<curr_program>
```

## Performance Results and Feedback:
The current cache eviction algorithm was evaluated on multiple real-world traces with different access patterns. Here are the results:

<inputs_outputs_feedback>

## Key Information:
- Higher hit rates (lower miss rates) are better.
- `evict` defines how the algorithm chooses the eviction victim.
- `update_after_hit` defines how the algorithm update the metadata it maintains immediately after a cache hit.
- `update_after_insert` defines how the algorithm updates the metadata it maintains immediately after inserting a new object into the cache.
- `update_after_evict` defines how the algorithm updates the metadata it maintains immediately after evicting the victim.
You have read-only access to these data and no access to any functions:
- An "object" represents the unit of a request, such as inserting an object into the cache or retrieving an object from the cache. Each object `obj` provides the following **read-only** attributes that you can reference:
    - `obj.key` (str): A string that uniquely identifies the object.
    - `obj.size` (int): A positive integer representing the size of the object in bytes.
- You can also reference the following **read-only** attributes provided by a cache snapshots `cache_snapshot`:
    - `cache_snapshot.cache` (dict): A dictionary containing the cached objects, where the keys are the objects' keys, and the values are the corresponding objects themselves.
    - `cache_snapshot.size` (int): A non-negative integer representing the current total size of the cache in bytes.
    - `cache_snapshot.capacity` (int): A positive integer representing the maximum allowed size of the cache in bytes.
    - `cache_snapshot.access_count` (int): The current total number of cache accesses. You can also use this to represent current time.
    - `cache_snapshot.hit_count` (int): The current total number of cache hits.
    - `cache_snapshot.miss_count` (int): The current total number of cache misses.

## Your Task:
Analyze the performance feedback and rewrite the cache eviction algorithm to:
1. Reduce overall cache miss rates
2. Make better decisions about which object to evict when the cache is full
3. Make better designs about which metadata to maintain
4. Improve metadata updates after cache hits, inserts, and evictions

Provide the complete improved cache eviction algorithm implementation in Python.

```python
# Your improved cache eviction algorithm here
```
"""

    input_keys = ["current_instruction_doc", "dataset_with_feedback"]
    output_keys = ["new_program"]

    @classmethod
    def prompt_renderer(cls, input_dict: dict[str, str]) -> str:
        def format_samples(samples):
            formatted = []
            for i, sample in enumerate(samples, 1):
                s = f"Example {i}:\n"
                for key, val in sample.items():
                    s += f"- {key}: {val}\n"
                formatted.append(s.strip())
            return "\n\n".join(formatted)

        prompt = cls.prompt_template
        prompt = prompt.replace("<curr_program>", input_dict["current_instruction_doc"])
        prompt = prompt.replace("<inputs_outputs_feedback>", format_samples(input_dict["dataset_with_feedback"]))
        
        # Log the full prompt if verbose logging is enabled
        if os.environ.get("GEPA_LOG_PROMPTS", "0") == "1":
            logger.info("="*80)
            logger.info("LLM PROMPT:")
            logger.info("="*80)
            logger.info(prompt)
            logger.info("="*80)
        
        return prompt


    @classmethod
    def _parse_full_rewrite(cls, llm_response: str, language: str = "python") -> Optional[str]:
        """
        Extract a full rewrite from an LLM response

        Args:
            llm_response: Response from the LLM
            language: Programming language

        Returns:
            Extracted code or None if not found
        """
        code_block_pattern = r"```" + language + r"\n(.*?)```"
        matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Fallback to any code block
        code_block_pattern = r"```(.*?)```"
        matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Fallback to plain text
        return llm_response

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        # Log the full LLM response if verbose logging is enabled
        if os.environ.get("GEPA_LOG_PROMPTS", "0") == "1":
            logger.info("="*80)
            logger.info("LLM RESPONSE:")
            logger.info("="*80)
            logger.info(lm_out)
            logger.info("="*80)
        
        # TODO(shu): just add some config.language for other languages
        new_program = cls._parse_full_rewrite(lm_out, language="python")

        return {"new_program": new_program}
