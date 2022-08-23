import random
import numpy as np

# include prediction in state
# store state and transition incrementally, adding history incrementally?
# store list of rewards to observe progression

class Qtable:
    def __init__(self, 
        learning_rate,
        discount_factor,
        history_size = 3,
        exploration=(0.7, 0.01, 0.0001)):
        """ Description of the Class """
        
        self.actions = ("R", "P", "S")
        self.size = (len(self.actions)**history_size, len(self.actions))
        self.reward_space = (0.4, 0, -0.2)
        
        self.lr = learning_rate
        self.df = discount_factor
        self.decay_rate = exploration[1]
        self.explore_minimum = exploration[2]
        self.prediction = random.choice(self.actions)
        
        self.q = np.zeros(self.size)
        self.explore_start = exploration[0]
        self.explore = exploration[0]
        self.rewards = [0]
    
    def __getitem__(self, key):
        return self.actions.index(key)
    
    @property
    def table(self):
        return self.q
    
    def position(self, *keys):
        rev = keys[::-1]
        n = len(self.actions)
        return sum([self[k]*n**i for i, k in enumerate(rev)])
    
    def compute_reward(self, prev):
        previous_play = self.actions.index(prev)
        predicted_play = self.actions.index(self.prediction)
        offset = (previous_play - predicted_play) % len(self.actions)
        reward = self.reward_space[offset]
        
        if reward > 0:
            self.lr = max(self.lr*.9, 0.2)
            self.df = max(self.df*.95, 0.2)
        else:
            self.lr = min(self.lr*1.05, 1.0)
            self.df = min(self.df*1.02, 1.0)
        
        return reward
    
    def evaluate(self, history):
        state = self.position(*history[:-1])
        next_state = self.position(*history[1:])
        reward = self.compute_reward(history[-1])
        return state, next_state, reward
    
    def bellman(self, alpha, gamma, reward, old_state, max_new_state):
        beta = 1 - alpha
        return beta*old_state + alpha*(reward + gamma*max_new_state)
    
    def update(self, history):
        self.explore = max(self.explore-self.decay_rate, self.explore_minimum)
        old, new, reward = self.evaluate(history)
        action = self.actions.index(self.prediction)
        self.rewards.append(reward)

        old_state = self.q[old, action]
        new_state = max(self.q[new, :])
        
        value = self.bellman(self.lr, self.df, reward, old_state, new_state)
        
        self.q[old, action] = value
        new_prediction = self.predict(new)
        self.prediction = new_prediction
        return new_prediction
    
    def predict(self, state):
        if random.random() < self.explore:
            return random.choice(self.actions)
        else:
            best_guess = np.argmax(self.q[state, :])
            return self.actions[best_guess]
    
    def reset(self):
        self.q = np.zeros(self.q.shape)
        self.explore = self.explore_start
        self.rewards = [0]
        self.prediction = "R"
    
    def __del__(self):
        print(self.rewards)
        print(np.round(self.q, 2))


def player_generator(lr, gamma, length, verbose=False):
    q = Qtable(lr, gamma, length)
    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}
    
    def inner(prev, history=[]):
        if not prev:
            history.clear()
            q.reset()
            history += random.choices(q.actions, k=length)
        
        if len(history) > length:
            _ = history.pop(0)
        
        history.append(prev if prev else "R")
        prediction = q.update(history)
        if verbose:
            print(sum(q.rewards))
        return ideal_response[prediction]
    return inner

player = player_generator(0.85, .9, 3)



