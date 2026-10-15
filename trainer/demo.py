"""Fixed, explicit-range river study position for the one-click trainer demo.

This is a single decision in a restricted game, not a hand reconstructed from
the playable table. The server always solves it afresh through the shared API.
"""

DEMO_RIVER_REQUEST = {
    "street": "river",
    "board": "As Kh Qd 7c 2s",
    "hero_hand": "8h 8s",
    "pot": 8,
    "stacks": [10, 10],
    "first_actor": 0,
    "iterations": 2000,
    "ranges": [
        {"hands": [["Jc", "Tc"], ["8h", "8s"]], "weights": [1, 1]},
        {"hands": [["9h", "9s"], ["Ac", "Ad"]], "weights": [1, 1]},
    ],
}

DEMO_DESCRIPTION = (
    "Fixed river decision: hero is the big blind and acts first. Pot is 8 chips; "
    "both effective stacks are 10 chips at the start of river betting. "
    "Each listed private combo has equal assumed prior weight within its seat. "
    "These ranges are illustrative and were not inferred from a played hand. "
    "The bet sizes and one-raise limit define a restricted game."
)
