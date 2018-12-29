# Game AI: Your starting point for AI research

> NOTE: This is still a work in progress and the API might change significantly before a stable release. Use at your own risk.

Welcome to gameai! This package contains a series of well-defined abstractions that are common in AI, such as a games, agents, and trainers which optimize the behavior of agents. As long as a class inherits from the base implemetation of a primitive (e.g. `Agent`) and implements the required methods, it can be used in place of the standard implementations given. To get started:

```python
from gameai.games import TicTacToe
from gameai.agents import RandomAgent, MCTSAgent
from gameai.core import Arena, Player

# Create our game
game = TicTacToe()

# Inititalize our agents
mcts_agent = MCTSAgent()
random_agent = RandomAgent()

# We train the mcts agent
mcts_agent.train(game, verbose=True, num_iters=10000, num_episodes=100)

player0 = Player(0, mcts_agent)
player1 = Player(1, random_agent)

# Pit the agents against eachother in the arena. Note that the player
# ids passed in need to match the index of the player in the array
arena = Arena(game, [player0, player1])
arena.play_games(verbose=True)
arena.statistics()
```

## Installation

Because this is still in alpha and under active development, it has not been released to PyPi. You can install via TestPyPi using the following command:

```bash
pip install --index-url https://test.pypi.org/simple/ gameai
```

## Running locally

Clone the repository and install the neccessary dependencies with `pip install -r requirements.txt`.

## Testing

Testing is done with `pytest`. Run the command `make test` to run all tests.

## Building

Build the project with `python3 setup.py sdist bdist_wheel`.

## Roadmap

- [x] Add Monte Carlo Tree Search
- [ ] Add Alpha Beta Pruning
- [ ] Add Othello as a game
- [ ] Add Chess as a game
- [ ] Add a simple RL agent & trainer
- [ ] Add an alphazero agent & trainer

## Contribution

Feel free to open an issue / submit a PR!
