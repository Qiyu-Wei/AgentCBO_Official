from __future__ import annotations
import os
from typing import Dict, List, Tuple

import numpy as np

from ..llm.client import LLMClient, PairwiseJudgment

_PAIRWISE_PROMPT = """You are a causal reasoning expert.
Domain context: {domain_knowledge}

Given two variables:
- A: {var_a}
- B: {var_b}

Determine the most likely causal relationship between A and B.
Consider direct causation, reverse causation, and the possibility of no direct link.

Respond with a JSON object:
{{"cause": "{var_a}", "effect": "{var_b}", "direction": "A->B" or "B->A" or "none", "confidence": 0.7, "reasoning": "brief explanation"}}"""


class LLMCausalPrior:

    def __init__(self, llm_client: LLMClient, strict: bool = None):
        self.llm = llm_client
        self._cache: Dict[Tuple[str, str, int], float] = {}
        if strict is None:
            strict = _env_truthy('AGENTCBO_STRICT_LLM_PRIOR') or _env_truthy('STRICT_LLM_PRIOR')
        self.strict = bool(strict)
        self._stats = {
            'pair_queries': 0,
            'cache_hits': 0,
            'structured_successes': 0,
            'structured_failures': 0,
            'raw_successes': 0,
            'raw_failures': 0,
            'defaulted_pairs': 0,
        }
        self._last_errors: List[str] = []

    def get_stats(self) -> Dict[str, object]:
        out = dict(self._stats)
        out['strict'] = self.strict
        out['last_errors'] = list(self._last_errors[-5:])
        out['llm_total_calls'] = getattr(self.llm, 'total_calls', None)
        out['llm_total_tokens'] = getattr(self.llm, 'total_tokens', None)
        return out

    def get_pairwise_prior(self, var_a: str, var_b: str,
                            domain_knowledge: str = None) -> float:
        dk_hash = hash(domain_knowledge) if domain_knowledge else 0
        cache_key = (var_a, var_b, dk_hash)
        if cache_key in self._cache:
            self._stats['cache_hits'] += 1
            return self._cache[cache_key]
        self._stats['pair_queries'] += 1

        prompt = _PAIRWISE_PROMPT.format(
            var_a=var_a, var_b=var_b,
            domain_knowledge=domain_knowledge or 'No specific domain context.')

        is_none = False
        try:
            result = self.llm.query_structured(prompt, PairwiseJudgment)
            self._stats['structured_successes'] += 1
            if result.direction == "A->B":
                score = max(0.5, min(1.0, result.confidence))
            elif result.direction == "B->A":
                score = min(0.5, max(0.0, 1.0 - result.confidence))
            else:
                score = 0.5 * (1.0 - result.confidence)
                is_none = True
        except Exception as structured_exc:
            self._record_error('structured', var_a, var_b, structured_exc)
            try:
                import json
                raw = self.llm.query(prompt)
                data = json.loads(raw)
                self._stats['raw_successes'] += 1
                # Accept a direction label or a numeric choice from the model.
                direction = data.get('direction')
                choice = data.get('choice')
                conf = float(data.get('confidence', 0.5))
                if direction is not None:
                    if direction == "A->B":
                        score = max(0.5, conf)
                    elif direction == "B->A":
                        score = min(0.5, 1.0 - conf)
                    else:
                        score = 0.5 * (1.0 - conf)
                        is_none = True
                elif choice is not None:
                    choice = int(choice)
                    if choice == 1:
                        score = max(0.5, conf)
                    elif choice == 2:
                        score = min(0.5, 1.0 - conf)
                    else:
                        score = 0.5 * (1.0 - conf)
                        is_none = True
                else:
                    score = 0.5
                    is_none = True
                    self._stats['defaulted_pairs'] += 1
                    if self.strict:
                        raise RuntimeError(
                            f"Strict LLM prior rejected schema-missing response for {var_a}/{var_b}"
                        )
            except Exception as raw_exc:
                self._record_error('raw', var_a, var_b, raw_exc)
                self._stats['defaulted_pairs'] += 1
                if self.strict:
                    raise RuntimeError(
                        f"Strict LLM prior failed for pair {var_a}/{var_b}: "
                        f"{type(raw_exc).__name__}: {raw_exc}"
                    ) from raw_exc
                score = 0.5
                is_none = True

        self._cache[cache_key] = score
        # "none": both directions unlikely → symmetric; directional → complementary
        self._cache[(var_b, var_a, dk_hash)] = score if is_none else 1.0 - score
        return score

    def get_full_prior_matrix(self, variable_names: List[str],
                               domain_knowledge: str = None) -> np.ndarray:
        n = len(variable_names)
        prior = np.full((n, n), 0.5)
        np.fill_diagonal(prior, 0.0)
        calls_before = getattr(self.llm, 'total_calls', None)
        for i in range(n):
            for j in range(i + 1, n):
                p_ij = self.get_pairwise_prior(
                    variable_names[i], variable_names[j], domain_knowledge)
                prior[i, j] = p_ij
                # Use cache which already has the correct reverse value
                reverse_key = (variable_names[j], variable_names[i],
                               hash(domain_knowledge) if domain_knowledge else 0)
                prior[j, i] = self._cache.get(reverse_key, 1.0 - p_ij)
        calls_after = getattr(self.llm, 'total_calls', None)
        if self.strict and n > 1 and calls_before == calls_after and not self._cache:
            raise RuntimeError('Strict LLM prior produced no LLM calls and no cache entries')
        return prior

    def _record_error(self, stage: str, var_a: str, var_b: str, exc: Exception) -> None:
        key = 'structured_failures' if stage == 'structured' else 'raw_failures'
        self._stats[key] += 1
        self._last_errors.append(f'{stage}:{var_a}/{var_b}:{type(exc).__name__}: {exc}')


def _env_truthy(name: str) -> bool:
    return os.getenv(name, '').strip().lower() in {'1', 'true', 'yes', 'on'}
