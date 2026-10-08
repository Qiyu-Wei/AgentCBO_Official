"""Published 37-node ALARM topology with a synthetic linear Gaussian SEM.

Supply the bnlearn model string through AGENTCBO_ALARM_MODEL_FILE.
The target is BP; interventions act on five operator-controlled settings.
"""
from __future__ import annotations
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..causal.dag import DAG, CausalGraph
from .base import CausalBenchmark

N_NODES, N_ARCS = 37, 46

TARGET = 'BP'
# Ventilator, oxygen, anaesthesia, and intubation settings.
MANIPULATIVE = ['FIO2', 'MVS', 'VMCH', 'ANES', 'INT']

DOMAIN_KNOWLEDGE = (
    "Intensive-care patient monitoring (the ALARM network). Variables are "
    "physiological states, ventilator and anaesthesia settings, and monitor "
    "readings. Abbreviations: HIST history, CVP central venous pressure, PCWP "
    "pulmonary capillary wedge pressure, HYP hypovolaemia, LVV left ventricular "
    "end-diastolic volume, LVF left ventricular failure, STKV stroke volume, ERLO "
    "error low output, HRBP heart rate from blood pressure, HREK heart rate from "
    "EKG, ERCA error cauter, HRSA heart rate from oxygen saturation, ANES "
    "insufficient anaesthesia or analgesia, APL anaphylaxis, TPR total peripheral "
    "resistance, ECO2 expired CO2, KINK kinked ventilator tube, MINV minute "
    "ventilation measured, FIO2 fraction of inspired oxygen, PVS pulmonary venous "
    "oxygen saturation, SAO2 arterial oxygen saturation, PAP pulmonary artery "
    "pressure, PMB pulmonary embolus, SHNT intrapulmonary shunt, INT intubation "
    "status, PRSS breathing pressure, DISC ventilator disconnection, MVS minute "
    "volume set on the ventilator, VMCH ventilator machine, VTUB ventilator tube, "
    "VLNG ventilator lung, VALV ventilator alveolus, ACO2 arterial CO2, CCHL "
    "catecholamine level, HR heart rate, CO cardiac output, BP arterial blood "
    "pressure."
)


def _parse(model_string: str) -> "OrderedDict[str, List[str]]":
    parents: "OrderedDict[str, List[str]]" = OrderedDict()
    for block in re.findall(r'\[([^\]]+)\]', model_string):
        if '|' in block:
            node, ps = block.split('|')
            parents[node] = ps.split(':')
        else:
            parents[block] = []
    return parents


def _topological(parents) -> List[str]:
    seen, order = set(), []
    while len(order) < len(parents):
        ready = sorted(n for n, ps in parents.items()
                       if n not in seen and all(p in seen for p in ps))
        if not ready:
            raise ValueError('ALARM model string is cyclic')
        for n in ready:
            seen.add(n)
            order.append(n)
    return order


class AlarmGraph(CausalBenchmark):
    """ALARM's published structure with linear additive-noise mechanisms."""

    def __init__(self, seed: int = 0, noise_scale: float = 0.3):
        self.seed = int(seed)
        self.noise_scale = float(noise_scale)
        model_file = os.environ.get('AGENTCBO_ALARM_MODEL_FILE')
        if not model_file:
            raise ValueError('Set AGENTCBO_ALARM_MODEL_FILE to a text file containing '
                             'the ALARM bnlearn model string; see README.md')
        self._parents = _parse(Path(model_file).read_text(encoding='utf-8'))
        if len(self._parents) != N_NODES:
            raise ValueError(f'expected {N_NODES} nodes, parsed {len(self._parents)}')
        arcs = sum(len(p) for p in self._parents.values())
        if arcs != N_ARCS:
            raise ValueError(f'expected {N_ARCS} arcs, parsed {arcs}')
        self._order = _topological(self._parents)
        rng = np.random.RandomState(self.seed)
        # Keep every structural edge's coefficient away from zero.
        self._w = {n: rng.choice([-1.0, 1.0], size=len(ps)) * rng.uniform(0.5, 1.5, size=len(ps))
                   for n, ps in self._parents.items()}

    def get_sem(self) -> OrderedDict:
        sem = OrderedDict()
        for name in self._order:
            ps, w = tuple(self._parents[name]), self._w[name]
            if not ps:
                sem[name] = lambda noise, pa: noise
            else:
                sem[name] = (lambda noise, pa, ps=ps, w=w:
                             sum(w[k] * pa[ps[k]] for k in range(len(ps))) + noise)
        return sem

    def get_ground_truth_dag(self) -> CausalGraph:
        names = list(self._order)
        idx = {n: i for i, n in enumerate(names)}
        adj = np.zeros((len(names), len(names)), dtype=int)
        for node, ps in self._parents.items():
            for p in ps:
                adj[idx[p], idx[node]] = 1
        return CausalGraph(DAG.from_adjacency_matrix(adj), names)

    def get_exploration_sets(self) -> List[Tuple[str, ...]]:
        return [(v,) for v in MANIPULATIVE]

    def get_intervention_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {v: (-3.0, 3.0) for v in MANIPULATIVE}

    def get_noise_distributions(self) -> Dict[str, Callable]:
        return {n: (lambda rng, s=self.noise_scale: rng.normal(0.0, s)) for n in self._order}

    @property
    def target_variable(self) -> str:
        return TARGET

    @property
    def manipulative_variables(self) -> List[str]:
        return list(MANIPULATIVE)

    @property
    def domain_knowledge(self) -> str:
        return DOMAIN_KNOWLEDGE
