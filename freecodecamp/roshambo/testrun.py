from RPS import player as rps_player
from RPS_game import play, kris, abbey
import numpy as np
import matplotlib.pyplot as plt
import functools


size = 10000
number_of_games = 10
parameters = np.arange(0, 1, 0.01)
wins = {}

opponent = abbey
w = [None] * len(parameters)
for i, discount in enumerate(parameters):
    player = functools.partial(rps_player, discount=discount)
    t = 0
    for _ in range(number_of_games):
        t += play(player, opponent, size)
    w[i] = t / number_of_games

w = np.array(w)
print(parameters[np.argmax(w)])    # .32
wins["abbey"] = w

opponent = kris
w = [None] * len(parameters)
for i, discount in enumerate(parameters):
    player = functools.partial(rps_player, discount=discount)
    t = 0
    for _ in range(number_of_games):
        t += play(player, opponent, size)
    w[i] = t / number_of_games

w = np.array(w)
print(parameters[np.argmax(w)])    # .84
wins["kris"] = w


plt.plot(parameters, wins["abbey"])
plt.show()

plt.plot(parameters, wins["kris"])
plt.show()


