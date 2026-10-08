from ..causal.dag import CausalGraph
from .base_cbo import BaseCBO


class OracleCBO(BaseCBO):
    """CBO supplied with the benchmark's ground-truth graph."""

    def _get_causal_graph(self) -> CausalGraph:
        return self.benchmark.get_ground_truth_dag()
