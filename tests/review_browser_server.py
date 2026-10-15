"""Deterministic real-game server fixture for the browser review test."""

import sys
from http.server import ThreadingHTTPServer

from poker_cpu.engine import HoldemGame
from trainer import server


class PassiveAgent:
    def act(self, observation):
        call = next((a for a in observation["legal_actions"] if a.kind == "call"), None)
        return call or next(a for a in observation["legal_actions"] if a.kind == "check")


class SeededSession(server.TableSession):
    def _new_hand(self, opponent):
        self.hand_number += 1
        self.opponent = "heuristic"
        self.action_log = []
        self.decision_reviews = []
        self.solver_reviews = []
        self.game = HoldemGame(button=0, seed=29)
        self.agent = PassiveAgent()
        self.adviser = None
        return self.state()


if __name__ == "__main__":
    server.TableSession = SeededSession
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), server.Handler).serve_forever()
