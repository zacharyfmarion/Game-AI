# Import the game that you want to play
from games.tictactoe import TicTacToe

# Import the agents you want to pit against eachother
from agents.random import RandomAgent
from agents.minimax import MinimaxAgent
from agents.mcts import MCTSAgent

# If you are using an agent that you need to train, import
# the trainer here
from trainers.mcts import MCTSTrainer

# We need the Arena to pit the players against eachother
from core.arena import Arena

# Create our game
game = TicTacToe()

# First train the mcts agent
print("Training agent...")
trainer = MCTSTrainer()
plays, wins = trainer.train(game, verbose=True)

# Inititalize our agents
print("\nPitting agents...\n")
agent1 = MCTSAgent(0, plays=plays, wins=wins)
agent2 = MinimaxAgent(1)
arena = Arena(game, [agent1, agent2])

arena.play_games()
arena.statistics()