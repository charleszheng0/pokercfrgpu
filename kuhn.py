import random

cards = {"J":11, "Q":12, "K":13}
p1_card, p2_card = random.sample(list(cards.keys()),2)
history = ""

#c = check
#b = bet
#f = fold
#k = call

def get_avail_actions(history):
    if history == "": 
        return ["check", "bet"]
    elif history == "c": 
        return ["check", "bet"]
    elif history == "b": 
        return ["fold", "call"]
    elif history == "cb":
        return ["fold", "call"]

def terminate(history): 
    if (history == "cc" 
    or history == "bf" 
    or history == "bk"
    or history == "cbf"
    or history == "cbk"
    ): 
        return True

while not terminate(history): 
    actions = get_avail_actions(history)
    print(actions)
    action = input("Choose action: ")
    
    if action == "check":
        history+='c'
    elif action == "bet":
        history+='b'
    elif action =="call":
        history+='k'
    elif action=="fold": 
        history+='f'

    print(action)

def payoff(history): 
    if history == "bf":
        return 1
    elif history =="cbf": 
        return -1
    else: 
        if cards[p1_card] > cards[p2_card]: 
            return 2
        else: 
            return -2



