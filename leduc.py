
import random


class LeducPoker:
    def __init__(self):
        # Initialize the game.
        self.reset()

    def reset(self):
        # Start a new hand with shuffled cards and fresh betting state.
        self.deck = ["J", "J", "Q", "Q", "K", "K"]
        random.shuffle(self.deck)

        self.players = [self.deck.pop(), self.deck.pop()]
        self.public_card = None

        self.pot = 2
        self.contributions = [1, 1]
        self.round_bets = [0, 0]

        self.round = 0
        self.current_player = 0
        self.raises_this_round = 0
        self.max_raises = 2

        self.done = False
        self.winner = None
        self.history = []
        self.actions_this_round = []

        return self.observe(self.current_player)

    def observe(self, player):
        # Return the information available to a player without revealing hidden cards.
        to_call = max(self.round_bets) - self.round_bets[player]

        return {
            "private_card": self.players[player],
            "public_card": self.public_card,
            "round": self.round,
            "pot": self.pot,
            "history": self.history.copy(),
            "to_call": to_call,
            "raises_remaining": self.max_raises - self.raises_this_round,
            "current_player": self.current_player,
            "round_bets": self.round_bets.copy(),
            "contributions": self.contributions.copy()
        }

    def legal_actions(self):
        # Determine legal moves from the current betting state.
        if self.done:
            return []

        player = self.current_player
        to_call = max(self.round_bets) - self.round_bets[player]

        if to_call == 0:
            return ["check", "bet"]

        actions = ["fold", "call"]

        if self.raises_this_round < self.max_raises:
            actions.append("raise")

        return actions

    def _pay(self, player, amount):
        # Move chips from a player into the pot.
        self.round_bets[player] += amount
        self.contributions[player] += amount
        self.pot += amount

    def _next_round(self):
        # Reveal the public card or finish the game after the final betting round.
        if self.round == 0:
            self.round = 1
            self.public_card = self.deck.pop()

            self.round_bets = [0, 0]
            self.raises_this_round = 0
            self.actions_this_round = []
            self.current_player = 0

        else:
            self.winner = self.showdown()
            self.done = True

    def step(self, action):
        # Apply one legal action and update the game's state.
        if self.done:
            raise ValueError("The game has already ended.")

        if action not in self.legal_actions():
            raise ValueError(
                f"Illegal action: {action}. "
                f"Legal actions: {self.legal_actions()}"
            )

        player = self.current_player
        opponent = 1 - player

        to_call = max(self.round_bets) - self.round_bets[player]
        bet_size = 2 if self.round == 0 else 4

        self.history.append((self.round, player, action))
        self.actions_this_round.append(action)

        if action == "fold":
            self.winner = opponent
            self.done = True

        elif action == "check":
            # Two consecutive checks complete the betting round.
            if self.actions_this_round[-2:] == ["check", "check"]:
                self._next_round()
            else:
                self.current_player = opponent

        elif action == "bet":
            self._pay(player, bet_size)
            self.current_player = opponent

        elif action == "call":
            self._pay(player, to_call)
            self._next_round()

        elif action == "raise":
            # A raise matches the outstanding amount plus one bet increment.
            self._pay(player, to_call + bet_size)
            self.raises_this_round += 1
            self.current_player = opponent

        return {
            "done": self.done,
            "winner": self.winner,
            "pot": self.pot,
            "next_player": None if self.done else self.current_player,
            "rewards": self.rewards() if self.done else [0, 0]
        }

    def showdown(self):
        # Determine the winner using pairs first, then high-card rank.
        ranks = {"J": 1, "Q": 2, "K": 3}

        p0, p1 = self.players

        pair0 = p0 == self.public_card
        pair1 = p1 == self.public_card

        if pair0 and not pair1:
            return 0

        if pair1 and not pair0:
            return 1

        if ranks[p0] > ranks[p1]:
            return 0

        if ranks[p1] > ranks[p0]:
            return 1

        return None

    def rewards(self):
        # Return each player's net chip winnings after the hand finishes.
        if not self.done:
            return [0, 0]

        if self.winner is None:
            return [
                self.pot / 2 - self.contributions[0],
                self.pot / 2 - self.contributions[1]
            ]

        rewards = [-self.contributions[0], -self.contributions[1]]
        rewards[self.winner] += self.pot

        return rewards


# ============================================
# PLAY A RANDOM GAME
# ============================================

game = LeducPoker()

while not game.done:
    player = game.current_player

    print(f"\nPlayer {player}'s turn")
    print("Observation:", game.observe(player))

    action = random.choice(game.legal_actions())

    print("Action:", action)
    game.step(action)

print("\n========== GAME OVER ==========")
print("Player cards:", game.players)
print("Public card:", game.public_card)
print("History:", game.history)
print("Pot:", game.pot)
print("Winner:", game.winner)
print("Rewards:", game.rewards())
