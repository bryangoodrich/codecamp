# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import random
import numpy as np

# Tracking 2 history states
observations = ("RR", "RP", "RS", "PR", "PP", "PS", "SR", "SP", "SS")
choices = ("R", "P", "S")
size = (len(observations), len(choices))
q = np.zeros(size)
last_play = ""

def player(prev_play, history = []):
    global last_play
    lr = 0.1
    gamma = 0.99
    explore = 1
    decay = 0.001
    min_explore = 0.01
    history.append(prev_play)
    
    # state needs to be -2:-1 and next_state is -2:
    reward = get_reward(last_play, prev_play)
    q[state, action] = (1-lr) * q[state, action] + lr*(reward + gamma*max(q[next_state,:]))
    
    if len(history) < 3:
        return random.choice(choices)
    
    rnd = random.random()
    obs = "".join(history[-2:])
    state = observations.index(obs)
    
    if rnd < explore:
        action = random.choice([0, 1, 2])
    else:
        action = choices[np.argmax(q[state, :])]
    
    explore = max(explore-decay, min_explore)
    return choices[action]




# Bellman Equation
# q[state, action] = (1-lr)*q[state, action] + lr*(r + g*max(q[state, :]))
# 