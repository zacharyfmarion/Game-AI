# Import the game that you want to play
from games.tictactoe import TicTacToe

# Import the agents you want to pit against eachother
from agents.minimax import MinimaxAgent

# We need the Arena and players to pit the agents against eachother
from core.arena import Arena
from core.player import Player


def play():
    # Create our game
    game = TicTacToe()

    # Inititalize our agents
    agent = MinimaxAgent()
    player1 = Player(0, agent)
    player2 = Player(1, agent)

    # Pit the agents against eachother in the arena. Note that the player
    # ids passed in need to match the index of the player in the array
    arena = Arena(game, [player1, player2])
    arena.play_game(verbose=True)


if __name__ == '__main__':
    play()
