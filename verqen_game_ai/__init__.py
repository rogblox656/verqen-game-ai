"""verqen-game-ai: plug-and-play neural game AI powered by VerQen.

Public API::

    from verqen_game_ai import VerQenAgent

    agent = VerQenAgent.from_pretrained(game="tictactoe")
    decision = agent.decide(board_state, available_actions)
"""

from .agent import GameDecision, VerQenAgent

__version__ = "1.0.0"

__all__ = ["GameDecision", "VerQenAgent", "__version__"]
