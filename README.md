# AI Games

This repository represents my attempts to replicate different AI algorithms to solve multiplayer zero-sum games. The `core/` folder contains the differnt classes that are used throughout the code such as `Game` and `Agent`. The `games/` folder is a collection of games that are subclassed from `Game`. Finally the `agents/` folder contains all the trained agents.

## Testing

Testing is done with `pytest`. Simply run the command `pytest` to run all tests.

## Roadmap

- [ ] Add Othello as a game
- [ ] Add Chess as a game
- [ ] Add Monte Carlo Tree Search
- [ ] Add a simple RL agent
- [ ] Add an alphazero agent

## Building

Build the project with `python3 setup.py sdist bdist_wheel`.

## Contribution

Feel free to open an issue / submit a PR!
