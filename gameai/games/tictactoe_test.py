from .tictactoe import TicTacToe


def test_action_space():
    game = TicTacToe()
    state = [-1, -1, 1, 0, -1, 0, 1, 0, -1]
    actions = game.action_space(state)
    expected = [0, 1, 4, 8]
    assert all([a == b for a, b in zip(actions, expected)])


def test_terminal():
    game = TicTacToe()
    state = [-1, -1, -1, 0, -1, 0, 1, 1, 1]
    assert game.terminal(state)
    state = [0, -1, -1, -1, 0, -1, -1, -1, 0]
    assert game.terminal(state)
    state = [1, 0, 1, 0, 1, 0, -1, -1, -1]
    assert not game.terminal(state)


def test_reward():
    game = TicTacToe()
    state = [-1, -1, -1, 0, -1, 0, 1, 1, 1]
    assert game.reward(state, 1) == 1
    assert game.reward(state, 0) == -1
    state = [0, -1, -1, -1, 0, -1, -1, -1, 0]
    assert game.reward(state, 1) == -1
    assert game.reward(state, 0) == 1
    state = [1, 0, 1, 0, 1, 0, -1, -1, -1]
    assert game.reward(state, 1) == 0
    assert game.reward(state, 0) == 0
