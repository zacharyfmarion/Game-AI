Getting Started
===============

Installation
------------

Because this is still in alpha and under active development, it has not been released to PyPi. You can install via TestPyPi using the following command:

.. code-block:: bash

  pip install --index-url https://test.pypi.org/simple/ gameai


Basic Example
----------------------

.. code-block:: python

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