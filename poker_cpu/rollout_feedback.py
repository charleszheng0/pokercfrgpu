"""Experimental full-hand action EVs against a named continuation policy.

The sampler replaces every hidden card before a copied game is played. It uses
only the hero's cards and public board from the current position; results are
not equilibrium values or a learned opponent-range estimate.
"""

from copy import deepcopy
from math import sqrt
import random
from time import perf_counter

from .agents import HeuristicAgent
from .cards import DECK


def estimate_action_evs(game, *, samples=24, seed=None, equity_samples=4):
    """Estimate net chips from hand start for each preset legal action.

    The first action is forced, then both seats use the same simple heuristic
    policy. Common sampled hidden deals are reused across actions.
    """
    if game.done or game.current_player != 0:
        raise ValueError("A hero decision is required for live EV estimation")
    if type(samples) is not int or not 2 <= samples <= 64:
        raise ValueError("Samples must be an integer from 2 to 64")
    if type(equity_samples) is not int or not 1 <= equity_samples <= 32:
        raise ValueError("Equity samples must be an integer from 1 to 32")

    started = perf_counter()
    actions = tuple(game.legal_actions())
    hero = tuple(game.observe(0)["hole"])
    board = tuple(game.observe(0)["board"])
    unseen = tuple(card for card in DECK if card not in set(hero + board))
    rng = random.Random(seed)
    outcomes = {action: [] for action in actions}

    for _ in range(samples):
        shuffled = list(unseen)
        rng.shuffle(shuffled)
        world = deepcopy(game)
        world.hole[1] = tuple(shuffled[:2])
        burn_count = len(world.burned)
        world.burned = list(shuffled[2:2 + burn_count])
        world.deck = shuffled[2 + burn_count:]
        policy_seed = rng.randrange(1 << 63)
        for action in actions:
            trial = deepcopy(world)
            trial.step(action)
            agents = (HeuristicAgent(policy_seed, samples=equity_samples),
                      HeuristicAgent(policy_seed + 1, samples=equity_samples))
            while not trial.done:
                seat = trial.current_player
                trial.step(agents[seat].act(trial.observe(seat)))
            outcomes[action].append(trial.rewards()[0])

    rows = []
    for action, values in outcomes.items():
        mean = sum(values) / samples
        variance = sum((value - mean) ** 2 for value in values) / (samples - 1)
        rows.append({"action": {"kind": action.kind, "amount": action.amount},
                     "ev_chips": mean, "ev_bb": mean / game.blinds[1],
                     "standard_error_chips": sqrt(variance / samples)})
    return {
        "status": "experimental_rollout",
        "street": ("preflop", "flop", "turn", "river")[game.street],
        "method": "Monte Carlo full-hand rollout",
        "assumptions": "Uniform hidden opponent hand; both seats use a sampled-equity heuristic after the displayed action; preset actions only.",
        "scope": "Uniform opponent hand and heuristic continuation; not GTO or the selected opponent's actual policy.",
        "samples_per_action": samples,
        "actions": rows,
        "latency_ms": (perf_counter() - started) * 1000,
    }
