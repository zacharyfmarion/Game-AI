# Import the game that you want to play
from games.tictactoe import TicTacToe

# Import the agents you want to pit against eachother
from agents.random import RandomAgent
from agents.minimax import MinimaxAgent
from agents.mcts import MCTSAgent

# If you are using an agent that you need to train, import
# the trainer here
from trainers.mcts import MCTSTrainer

# We need the Arena and players to pit the agents against eachother
from core.arena import Arena
from core.player import Player

# Create our game
game = TicTacToe()

# First train the mcts agent
print("Training agent...\n")
trainer = MCTSTrainer()
plays, wins = trainer.train(game, verbose=True)

# Inititalize our agents
print("\nPitting agents...\n")
player1 = Player(0, MCTSAgent(plays=plays, wins=wins))
player2 = Player(1, MinimaxAgent())

# Pit the agents against eachother in the arena. Note that the player
# ids passed in need to match the index of the player in the array
arena = Arena(game, [player1, player2])
arena.play_games()
arena.statistics()