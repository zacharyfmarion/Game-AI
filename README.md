# AI Games

This repository represents my attempts to replicate different AI algorithms to solve multiplayer zero-sum games. The `core/` folder contains the differnt classes that are used throughout the code such as `Game` and `Agent`. The `games/` folder is a collection of games that are subclassed from `Game`. Finally the `agents/` folder contains all the trained agents.

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
