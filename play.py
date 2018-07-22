from games.tictactoe import TicTacToe
from agents.random import RandomAgent
from agents.minimax import MinimaxAgent
from core.arena import Arena

# Inititalize our game and agents
game = TicTacToe()
agents = [MinimaxAgent(0), RandomAgent(1)]
arena = Arena(game, agents)

arena.play_games()
arena.statistics()