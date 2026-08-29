"""A small coding agent, built to be the control condition for research.

Deliberately minimal: a tool-calling loop, a contained workspace, full trajectory
recording with cost. Anything smarter lives in `askoract/` as a policy that can
be switched off and measured against this baseline.
"""
from .llm import LLMBackend, LLMResponse, Usage, make_backend
from .loop import Agent, AgentConfig, AgentResult, cli_responder
from .tools import AskUser, Finish, Tool, build_toolset
from .trace import Trace
from .workspace import Workspace

__all__ = [
    "Agent", "AgentConfig", "AgentResult", "cli_responder",
    "LLMBackend", "LLMResponse", "Usage", "make_backend",
    "AskUser", "Finish", "Tool", "build_toolset",
    "Trace", "Workspace",
]
