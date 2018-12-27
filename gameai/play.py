# Import the game that you want to play
from games import TicTacToe

# Import the agents you want to pit against eachother
from agents import RandomAgent, MCTSAgent

# If you are using an agent that you need to train, import
# the trainer here
from trainers import MCTSTrainer

# We need the Arena and players to pit the agents against eachother
from core import Arena, Player


def play():
    # Create our game
    game = TicTacToe()

    # First train the mcts agent
    print("Training agent...\n")
    trainer = MCTSTrainer(verbose=True, num_iters=10000, num_episodes=100)
    plays, wins = trainer.train(game)

    # Inititalize our agents
    print("\nPitting agents...\n")
    player0 = Player(0, MCTSAgent(plays=plays, wins=wins))
    player1 = Player(1, RandomAgent())

    # Pit the agents against eachother in the arena. Note that the player
    # ids passed in need to match the index of the player in the array
    arena = Arena(game, [player0, player1])
    arena.play_games(verbose=True)
    arena.statistics()


if __name__ == '__main__':
    play()
