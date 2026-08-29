"""Ask-or-Act: deciding when a coding agent should interrupt the user.

Pipeline:

    sample k candidates  ->  execute on probes  ->  cluster by behaviour
                         ->  score uncertainty  ->  gate  ->  localised question

Each stage is a separate module so any of them can be replaced by a control and
measured. `signals` in particular holds every baseline behind one interface, so
the gate never knows which signal it is holding.
"""
from .bse import BSEResult, ProbeDisagreement, compute_bse, normalised_entropy
from .execute import BehaviourMatrix, run_candidates
from .policy import AskOrActPolicy, Decision
from .probes import filter_probes, synthesize_probes
from .question import Question, build_question, next_question
from .signals import (BSESignal, ConstantSignal, DistinctClassesSignal,
                      MaxProbeEntropySignal, RandomSignal, SelfReportSignal,
                      Signal, SignalInput, TextDiversitySignal, TokenEntropySignal,
                      build_signal_input, offline_signals)

__all__ = [
    "BSEResult", "ProbeDisagreement", "compute_bse", "normalised_entropy",
    "BehaviourMatrix", "run_candidates",
    "AskOrActPolicy", "Decision",
    "filter_probes", "synthesize_probes",
    "Question", "build_question", "next_question",
    "Signal", "SignalInput", "BSESignal", "MaxProbeEntropySignal",
    "DistinctClassesSignal", "TextDiversitySignal", "TokenEntropySignal",
    "SelfReportSignal", "RandomSignal", "ConstantSignal",
    "build_signal_input", "offline_signals",
]
