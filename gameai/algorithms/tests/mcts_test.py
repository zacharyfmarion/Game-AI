import pytest

from ..mcts import MCTS
from ...games import LineTacToe


class FakeGame:
    total_action_space_size = 3

    def to_hash(self, _):
        return 1


def test_policy_assertion():
    with pytest.raises(ValueError):
        mcts = MCTS()
        game = LineTacToe()
        state = [-1, -1, -1]
        mcts.policy(game, state)


def test_policy_zero_temp():
    mcts = MCTS()
    mcts.plays = {
        (1, 0): 4
    }
    mcts.wins = {
        (1, 0): 2
    }
    game = FakeGame()
    state = [-1, -1, -1]
    expected = [1, 0, 0]
    policy = mcts.policy(game, state, temp=0)
    print(policy)
    for a, b in zip(expected, policy):
        assert a == b


def test_policy_full_temp():
    mcts = MCTS()
    mcts.plays = {
        (1, 0): 4,
        (1, 1): 4
    }
    mcts.wins = {
        (1, 0): 2,
        (1, 1): 2
    }
    game = FakeGame()
    state = [-1, -1, -1]
    expected = [0.5, 0.5, 0]
    policy = mcts.policy(game, state)
    for a, b in zip(expected, policy):
        assert a == b
