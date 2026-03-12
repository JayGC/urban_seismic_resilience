"""
RQ4 Strategic Horizon Analysis with LLM Commander.

Compares hierarchical (LLM Commander = global/strategic) vs decentralized (greedy)
in trap scenario. Answers: Do agents exhibit Second-Order Thinking vs Greedy Steps?
"""

from rq4_llm.experiment import run_rq4_llm_experiment
from rq4_llm.analysis import run_rq4_llm_analysis

__all__ = ['run_rq4_llm_experiment', 'run_rq4_llm_analysis']
