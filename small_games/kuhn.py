"""Original tabular CFR experiment for three-card Kuhn poker."""

cards = {"J": 11, "Q": 12, "K": 13}

# c = check
# b = bet
# f = fold
# k = call


def get_avail_actions(history):
    if history == "":
        return ["check", "bet"]
    elif history == "c":
        return ["check", "bet"]
    elif history == "b":
        return ["fold", "call"]
    elif history == "cb":
        return ["fold", "call"]


def action_to_char(action):
    if action == "check":
        return "c"
    elif action == "bet":
        return "b"
    elif action == "fold":
        return "f"
    elif action == "call":
        return "k"


def terminate(history):
    if (
        history == "cc"
        or history == "bf"
        or history == "bk"
        or history == "cbf"
        or history == "cbk"
    ):
        return True


def payoff(history, p1_card, p2_card):
    if history == "bf":
        return 1
    elif history == "cbf":
        return -1
    else:
        if cards[p1_card] > cards[p2_card]:
            if history == "cc":
                return 1
            return 2
        else:
            if history == "cc":
                return -1
            return -2


# regrets measures how much you wish you had chosen a different action at a decision point
# at terminal payoff points, those are used to calculate regret at the decision that led to it
# each ACTION has its regret value

regrets = {
    "J": {"check": 0, "bet": 0},
    "Q": {"check": 0, "bet": 0},
    "K": {"check": 0, "bet": 0},
    "Jc": {"check": 0, "bet": 0},
    "Qc": {"check": 0, "bet": 0},
    "Kc": {"check": 0, "bet": 0},
    "Jb": {"fold": 0, "call": 0},
    "Qb": {"fold": 0, "call": 0},
    "Kb": {"fold": 0, "call": 0},
    "Jcb": {"fold": 0, "call": 0},
    "Qcb": {"fold": 0, "call": 0},
    "Kcb": {"fold": 0, "call": 0},
}

strategy_sum = {  # sum of all strategy's reach weighted EVs.
    "J": {"check": 0, "bet": 0},
    "Q": {"check": 0, "bet": 0},
    "K": {"check": 0, "bet": 0},
    "Jc": {"check": 0, "bet": 0},
    "Qc": {"check": 0, "bet": 0},
    "Kc": {"check": 0, "bet": 0},
    "Jb": {"fold": 0, "call": 0},
    "Qb": {"fold": 0, "call": 0},
    "Kb": {"fold": 0, "call": 0},
    "Jcb": {"fold": 0, "call": 0},
    "Qcb": {"fold": 0, "call": 0},
    "Kcb": {"fold": 0, "call": 0},
}


def cfr(history, p1_card, p2_card, p1_reach, p2_reach):
    if terminate(history):
        return payoff(history, p1_card, p2_card)
    else:
        if len(history) % 2 == 0:
            infoset = p1_card + history  # "Qcb"
        else:
            infoset = p2_card + history  # "Qcbk" -> the stuff we look up
        regret_actions = regrets[infoset]  # values of regrets dictionary
        poss1, poss2 = get_avail_actions(
            history
        )  # available actions based on current history
        prop1, prop2 = regretMatch(
            regret_actions[poss1], regret_actions[poss2]
        )  # RM's proportions

        # only update p1 when p1's turn, likewise for p2
        if len(history) % 2 == 0:
            value1 = cfr(
                history + action_to_char(poss1),
                p1_card,
                p2_card,
                p1_reach * prop1,
                p2_reach,
            )  # cfr(key1) = p1's regret value
            value2 = cfr(
                history + action_to_char(poss2),
                p1_card,
                p2_card,
                p1_reach * prop2,
                p2_reach,
            )  # cfr(key2)

        else:
            value1 = cfr(
                history + action_to_char(poss1),
                p1_card,
                p2_card,
                p1_reach,
                p2_reach * prop1,
            )
            value2 = cfr(
                history + action_to_char(poss2),
                p1_card,
                p2_card,
                p1_reach,
                p2_reach * prop2,
            )
        curr_ev = prop1 * value1 + prop2 * value2  # EV under current strategy
        if len(history) % 2 == 0:
            regret_actions[poss1] += p2_reach * (
                value1 - curr_ev
            )  # update regret dictionary
            regret_actions[poss2] += p2_reach * (value2 - curr_ev)

            strategy_sum[infoset][poss1] += p1_reach * prop1
            strategy_sum[infoset][poss2] += p1_reach * prop2

        else:
            # p2's regret needs to be reversed since all values are p1's
            regret_actions[poss1] += p1_reach * (
                curr_ev - value1
            )  # update regret dictionary
            regret_actions[poss2] += p1_reach * (curr_ev - value2)

            strategy_sum[infoset][poss1] += p2_reach * prop1
            strategy_sum[infoset][poss2] += p2_reach * prop2

    return curr_ev  # -> acts as payoff, its just that we must return up the calculated value via both nodes


def regretMatch(val1, val2):
    if val1 > 0 and val2 > 0:
        val1Proportion = val1 / (val1 + val2)
        val2Proportion = val2 / (val1 + val2)
    elif val1 > 0 and val2 <= 0:
        val1Proportion = 1
        val2Proportion = 0
    elif val2 > 0 and val1 <= 0:
        val1Proportion = 0
        val2Proportion = 1
    else:
        val1Proportion = 0.5
        val2Proportion = 0.5
    return val1Proportion, val2Proportion


def train(iterations=12342):
    """Run the preserved Kuhn CFR experiment explicitly."""
    card_faces = cards.keys()
    for _ in range(iterations):
        for card1 in card_faces:
            for card2 in card_faces:
                if card1 != card2:
                    cfr("", card1, card2, 1.0, 1.0)

    nash_strategies = {}
    for infoset in strategy_sum:
        total = sum(strategy_sum[infoset].values())
        nash_strategies[infoset] = {}
        if total > 0:
            for action in strategy_sum[infoset]:
                nash_strategies[infoset][action] = strategy_sum[infoset][action] / total
    return nash_strategies


if __name__ == "__main__":
    print(train())
