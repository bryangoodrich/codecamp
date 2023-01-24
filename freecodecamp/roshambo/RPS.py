import random
import numpy as np
import itertools


DOUBLEQTABLES = True
DISCOUNT_FACTOR = 0.32
DISCOUNT_RATE = 0.3
DECAY_RATE = 0.5
LEARN_RATE = 0.8
HISTORY_SIZE = 4
IDEAL_RESPONSE = {'P': 'S', 'R': 'P', 'S': 'R'}
STATE_SIZE = (3**HISTORY_SIZE, 3)
STATES = ["".join(i) for i in itertools.product("RPS", repeat=HISTORY_SIZE)]


def get_reward(action, actual):
    prediction = "RPS"[action]
    if prediction == actual:
        return 1
    elif prediction == IDEAL_RESPONSE[actual]:
        return -1
    else:
        return -1


def map_state(state):
    s = "".join(state)
    return STATES.index(s)


def initialize_qtable(size):
    #return np.random.uniform(size=size)
    return np.zeros(size)


def maxQA(q, s):
    state = q[s]
    return np.argmax(state), max(state)  # (action, value)


def prediction(action):
    prediction = "RPS"[action]
    return IDEAL_RESPONSE[prediction]


qtablea = initialize_qtable(STATE_SIZE)
qtableb = initialize_qtable(STATE_SIZE)
action = random.choice([0, 1, 2])
counter = {}


def player(prev, history = [], discount=.84):
    global qtablea
    global qtableb
    global action
    global counter
    
    if prev == "":
        qtablea = initialize_qtable(STATE_SIZE)
        qtableb = initialize_qtable(STATE_SIZE)
        action = random.choice([0, 1, 2])
        counter = {}
        history.clear()
    
    if len(history) <= HISTORY_SIZE:
        history.append(prev)
        action = random.choice([0, 1, 2])
        return prediction(action)
    
    s1 = history[-HISTORY_SIZE:]
    state = map_state(s1)
    history.append(prev)
    s2 = history[-HISTORY_SIZE:]
    new_state = map_state(s2)
    reward = get_reward(action, prev)
    
    counter[state] = counter.get(state, 0) + 1
    counter[(state, action)] = counter.get((state, action), 0) + 1
    learning_rate = 1 / np.power(counter.get((state, action), 1), LEARN_RATE)
    discount = 1 / np.power(counter.get(state, 1), DISCOUNT_RATE)
    
    if DOUBLEQTABLES:
        if (np.random.random() < 0.5):  # Update A
            next_action, _ = maxQA(qtablea, new_state)
            current_q = qtablea[state, action]
            future_q = qtableb[new_state, next_action]
            discount = 1 - (learning_rate)
            new_q = (1-learning_rate)*current_q + learning_rate * (reward + discount*future_q)
            qtablea[state, action] = new_q
        else: # Update B
            next_action, _ = maxQA(qtableb, new_state)
            current_q = qtableb[state, action]
            future_q = qtablea[new_state, next_action]
            discount = 1 - (learning_rate)
            new_q = (1-learning_rate)*current_q + learning_rate * (reward + discount*future_q)
            qtableb[state, action] = new_q
    else:  # Single Q-Learning
        next_action, maxq = maxQA(qtablea, new_state)
        current_q = qtablea[state, action]
        future_q = maxq 
        new_q = (1-learning_rate)*current_q + learning_rate*(reward + discount*future_q)
        qtablea[state, action] = new_q
    
    eps = 1 / np.power(counter.get(state, 1), DECAY_RATE)
    action = next_action if np.random.random() < (1-eps) else random.choice([0, 1, 2])
    return prediction(action)
