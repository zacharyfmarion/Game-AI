import pytest

from ...games import TicTacToe
from ...agents import RandomAgent
from ..player import Player
from ..arena import Arena


def basic_arena():
    game = TicTacToe()
    player0 = Player(0, RandomAgent())
    player1 = Player(1, RandomAgent())
    arena = Arena(game, [player0, player1])
    return arena


def test_empty_statistics():
    arena = basic_arena()
    with pytest.raises(ZeroDivisionError):
        arena.statistics()


def test_basic_statistics(capsys):
    arena = basic_arena()
    arena.wins = [1, 1]
    arena.games_played = 2
    arena.statistics()
    out, _ = capsys.readouterr()
    assert out == 'Player 0: \n  - Games: 1 / 2\n  - Percentage: 50.0%\nPlayer 1: \n  - Games: 1 / 2\n  - Percentage: 50.0%\n'


def test_play_game():
    arena = basic_arena()
    winner = arena.play_game()
    assert arena.games_played == 1
    assert winner in [0, 1, -1]
