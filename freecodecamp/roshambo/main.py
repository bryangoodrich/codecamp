# This entrypoint file to be used in development. Start by reading README.md
from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from RPS import player
from unittest import main

plays = 1000
print("Player2 - quincy")
play(player, quincy, plays)

print("Player2 - mrugesh")
play(player, mrugesh, plays)

print("Player2 - kris")
play(player, kris, plays)

print("Player2 - abbey")
play(player, abbey, plays)

# Uncomment line below to play interactively against a bot:
#play(human, abbey, 20, verbose=True)

# Uncomment line below to play against a bot that plays randomly:
# play(human, random_player, 1000)


# Uncomment line below to run unit tests automatically
# main(module='test_module', exit=False)


