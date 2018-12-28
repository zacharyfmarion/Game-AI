# Game AI: Your starting point for AI research

Welcome to gameai! This package conatains a series of well-defined abstractions that are common in AI, such as a games, agents, and trainers which optimize the behavior of agents. As long as a class inherits from the bas implemetation of a primitive (e.g. `Agent`) and implements the required methods, it can be used in place of the standard implementations given. To get started:

```python
from gameai.games import TicTacToe
from gameai.agents import RandomAgent, MCTSAgent
from gameai.trainers import MCTSTrainer
from gameai.core import Arena, Player

# Create our game
game = TicTacToe()

# First train the mcts agent
trainer = MCTSTrainer(verbose=True, num_iters=10000, num_episodes=100)
plays, wins = trainer.train(game)

# Inititalize our agents
player0 = Player(0, MCTSAgent(plays=plays, wins=wins))
player1 = Player(1, RandomAgent())

# Pit the agents against eachother in the arena. Note that the player
# ids passed in need to match the index of the player in the array
arena = Arena(game, [player0, player1])
arena.play_games(verbose=True)
arena.statistics()
```

## Installation

```bash
pip install gameai
```

## Running locally

Clone the repository and install the neccessary dependencies with `pip install -r requirements.txt`.

## Testing

Testing is done with `pytest`. Simply run the command `pytest` to run all tests.

## Building

Build the project with `python3 setup.py sdist bdist_wheel`.

## Roadmap

- [x] Add Monte Carlo Tree Search
- [ ] Add Othello as a game
- [ ] Add Chess as a game
- [ ] Add a simple RL agent & trainer
- [ ] Add an alphazero agent & trainer

## Contribution

Feel free to open an issue / submit a PR!
