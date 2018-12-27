from gameai.games import TicTacToe
from gameai.agents import MinimaxAgent
from gameai.core import Arena, Player


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
    arena.play_games(verbose=True)
    arena.statistics()


if __name__ == '__main__':
    play()
