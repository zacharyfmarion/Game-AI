from games.tictactoe import TicTacToe
from agents.minimax import MinimaxAgent

# Inititalize our game, state, and current player
game = TicTacToe()
agent = MinimaxAgent()
state, player = game.initial_state()

# Play out the full game
while not game.terminal(state):
  action = agent.action(game, state, player)
  state[action] = player
  print(game.to_string(state), "\n")
  player = 1 - player