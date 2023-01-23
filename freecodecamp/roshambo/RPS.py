import random
import numpy as np
import itertools
from collections import Counter


EXPLORE_RATE = 0.99
DECAY_RATE = 0.01
EXPLORE_FLOOR = 0.001

LEARNING_RATE_MAX = 0.3
LEARNING_RATE = 0.1
LEARNING_DELTA = 0.005

DISCOUNT_FACTOR = 0.99
DISCOUNT_MIN = 0.8
DISCOUNT_DELTA = 0.005

HISTORY_SIZE = 4

IDEAL_RESPONSE = {'P': 'S', 'R': 'P', 'S': 'R'}
RPS = "RPS"
STATE_SIZE = (3**HISTORY_SIZE, len(RPS))
STATES = ["".join(i) for i in itertools.product("RPS", repeat=HISTORY_SIZE)]


def get_reward(prediction, actual):
    if prediction == actual:
        return 1
    elif prediction == IDEAL_RESPONSE[RPS[actual]]:
        return -1
    else:
        return -0.5


def get_new_action(options, explore_rate):
    r = random.random()
    action = random.choice([0, 1, 2]) if r < explore_rate else np.argmax(options)
    return action, max(explore_rate - DECAY_RATE, EXPLORE_FLOOR)


def map_state(s):
    state = "".join([RPS[i] for i in s])
    return STATES.index(state)


def initialize_qtable(size):
    return np.random.uniform(size=size)


def reset_counter(counter):
    for key in counter:
        counter[key] = 0
    
    return counter


# module variables
qtable = initialize_qtable(STATE_SIZE)
explore_rate = EXPLORE_RATE
discount = DISCOUNT_FACTOR
learning_rate = LEARNING_RATE
action = random.choice([0,1,2])
counter = reset_counter(Counter(STATES))


def player(previous_opponent_play, history = []):
    global qtable
    global explore_rate
    global learning_rate
    global discount
    global action
    global counter
    
    if previous_opponent_play == "":
        history.clear()
        qtable = initialize_qtable(STATE_SIZE)
        explore_rate = EXPLORE_RATE
        learning_rate = LEARNING_RATE
        discount = DISCOUNT_FACTOR
        action = random.choice([0, 1, 2])
        counter = reset_counter(counter)
    
    obs = RPS.index(previous_opponent_play)
    if len(history) <= HISTORY_SIZE:
        history.append(obs)
        return random.choice("RPS")
    
    s1 = history[-HISTORY_SIZE:]
    state = map_state(s1)
    history.append(obs)
    s2 = history[-HISTORY_SIZE:]
    new_state = map_state(s2)
    reward = get_reward(action, obs)
    
    max_future_q = np.max(qtable[new_state])
    current_q = qtable[state][action]
    new_q = max(0.0001, (1-learning_rate)*current_q + learning_rate*(reward + discount*max_future_q))
    qtable[state][action] = new_q
    qtable[state] = qtable[state] / sum(qtable[state])  # normalize to probability
    learning_rate = min(learning_rate + LEARNING_DELTA, LEARNING_RATE_MAX)
    discount = max(discount - DISCOUNT_DELTA, DISCOUNT_MIN)
    
    action, explore_rate = get_new_action(qtable[new_state], explore_rate)
    return IDEAL_RESPONSE[RPS[action]]
