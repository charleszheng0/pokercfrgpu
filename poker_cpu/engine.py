"""Heads-up hold'em with a small, explicit no-limit action abstraction.

Action.amount is the target contribution on the current street for bet/raise.
Stacks reset per hand. The caller rotates button between hands.
"""

from dataclasses import dataclass
import random

from .cards import DECK, evaluate, parse_cards


@dataclass(frozen=True)
class Action:
    kind: str
    amount: int = 0


class HoldemGame:
    def __init__(self, stacks=(100, 100), blinds=(1, 2), button=0, seed=None, deck=None):
        if len(stacks) != 2 or min(stacks) < blinds[1] or blinds[0] <= 0 or blinds[1] <= blinds[0]:
            raise ValueError("Use two stacks at least the big blind and valid blinds")
        if button not in (0, 1):
            raise ValueError("button must be 0 or 1")
        self.initial_stacks = tuple(stacks)
        self.stacks = list(stacks)
        self.blinds = tuple(blinds)
        self.button = button
        self.rng = random.Random(seed)
        if deck is None:
            self.deck = list(DECK)
            self.rng.shuffle(self.deck)
        else:
            self.deck = list(parse_cards(deck))
            if len(self.deck) != 52:
                raise ValueError("A scripted deck needs all 52 cards")
        # Deal alternately, starting at the button. deck[0] is the next card.
        self.hole = [[], []]
        for _ in range(2):
            for p in (button, 1 - button):
                self.hole[p].append(self.deck.pop(0))
        self.hole = [tuple(h) for h in self.hole]
        self.board = []
        self.street = 0
        self.paid = [0, 0]
        self.street_paid = [0, 0]
        self.pot = 0
        self.history = []
        self.done = False
        self.winner = None
        self.last_full_raise = blinds[1]
        self.raises_open = True
        self.pending = 1
        self.current_player = button
        self._put(button, blinds[0])
        self._put(1 - button, blinds[1])

    def _put(self, p, chips):
        assert 0 <= chips <= self.stacks[p]
        self.stacks[p] -= chips
        self.paid[p] += chips
        self.street_paid[p] += chips
        self.pot += chips

    def observe(self, p):
        """Only public state and this player's own cards; terminal cards stay private."""
        return {
            "player": p, "hole": self.hole[p], "board": tuple(self.board),
            "street": self.street, "button": self.button,
            "stacks": tuple(self.stacks), "pot": self.pot,
            "paid": tuple(self.paid), "street_paid": tuple(self.street_paid),
            "to_call": max(self.street_paid) - self.street_paid[p],
            "history": tuple(self.history),
            "legal_actions": tuple(self.legal_actions()) if p == self.current_player else (),
        }

    def legal_actions(self):
        if self.done:
            return []
        p = self.current_player
        q = 1 - p
        owed = max(self.street_paid) - self.street_paid[p]
        actions = [Action("fold"), Action("call")] if owed else [Action("check")]
        if not self.stacks[p] or not self.stacks[q]:
            return actions
        # Chips beyond the opponent's effective stack cannot be wagered heads-up.
        cap = min(self.street_paid[p] + self.stacks[p], self.street_paid[q] + self.stacks[q])
        highest = max(self.street_paid)
        if cap <= highest or (owed and not self.raises_open):
            return actions
        min_target = highest + self.last_full_raise
        if cap < min_target:
            targets = {cap}  # Legal short all-in raise.
        else:
            pot_after_call = self.pot + owed
            targets = {min_target, min(cap, highest + max(self.last_full_raise, pot_after_call // 2)),
                       min(cap, highest + max(self.last_full_raise, pot_after_call)), cap}
        kind = "raise" if owed else "bet"
        return actions + [Action(kind, t) for t in sorted(targets) if t > highest]

    def _runout(self):
        while self.street < 3:
            self.street += 1
            self._reveal()
        self._showdown()

    def _reveal(self):
        # Standard burn before flop, turn, and river.
        self.deck.pop(0)
        count = 3 if self.street == 1 else 1
        self.board.extend(self.deck[:count])
        del self.deck[:count]

    def _next_street(self):
        if self.street == 3:
            self._showdown()
            return
        self.street += 1
        self._reveal()
        self.street_paid = [0, 0]
        self.last_full_raise = self.blinds[1]
        self.raises_open = True
        self.pending = 2
        self.current_player = 1 - self.button
        if min(self.stacks) == 0:
            self._runout()

    def _award(self, winner):
        self.done = True
        self.winner = winner
        if winner is None:
            assert self.pot % 2 == 0
            self.stacks[0] += self.pot // 2
            self.stacks[1] += self.pot // 2
        else:
            self.stacks[winner] += self.pot

    def _showdown(self):
        a = evaluate(self.hole[0] + tuple(self.board))
        b = evaluate(self.hole[1] + tuple(self.board))
        self._award(0 if a > b else 1 if b > a else None)

    def step(self, action):
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action}; legal: {self.legal_actions()}")
        p = self.current_player
        q = 1 - p
        owed = max(self.street_paid) - self.street_paid[p]
        self.history.append((self.street, p, action.kind, action.amount))
        if action.kind == "fold":
            self._award(q)
        elif action.kind == "check":
            self.pending -= 1
            if self.pending == 0:
                self._next_street()
            else:
                self.current_player = q
        elif action.kind == "call":
            self._put(p, min(owed, self.stacks[p]))
            # If the caller is shorter, return the uncalled part of the bet.
            excess = self.paid[q] - self.paid[p]
            if excess > 0:
                self.paid[q] -= excess
                self.street_paid[q] -= excess
                self.stacks[q] += excess
                self.pot -= excess
            if self.street == 0 and len(self.history) == 1 and p == self.button and min(self.stacks) > 0:
                # A completed small blind still leaves the big blind its option.
                self.current_player = q
                self.pending = 1
            else:
                self._next_street()
        else:
            highest = max(self.street_paid)
            increment = action.amount - highest
            self._put(p, action.amount - self.street_paid[p])
            self.raises_open = increment >= self.last_full_raise
            if self.raises_open:
                self.last_full_raise = increment
            self.pending = 1
            self.current_player = q
        return self.observe(self.current_player) if not self.done else self.rewards()

    def rewards(self):
        if not self.done:
            raise ValueError("Hand is not terminal")
        return tuple(self.stacks[p] - self.initial_stacks[p] for p in (0, 1))
