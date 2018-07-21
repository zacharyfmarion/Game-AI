from games.tictactoe import TicTacToe
from agents.limited_depth_minimax import LimitedDepthMinimaxAgent
from agents.random import RandomAgent

# Inititalize our game and agents
game = TicTacToe()
agent1 = RandomAgent(0)
agent2 = LimitedDepthMinimaxAgent(1)

state = game.initial_state()
curr_agent = agent1

# Play out the full game
while not game.terminal(state):
  action = curr_agent.action(game, state)
  state[action] = int(curr_agent)
  print(game.to_readable_string(state), "\n")
  curr_agent = agent2 if curr_agent == agent1 else agent1