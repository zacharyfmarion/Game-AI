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

  from games import TicTacToe
  from agents import RandomAgent, MCTSAgent
  from core import Arena, Player

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