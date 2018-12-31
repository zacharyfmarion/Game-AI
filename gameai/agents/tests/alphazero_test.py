from ..alphazero_agent import AlphaZeroAgent


class FakeNetwork:
    def predict_single(self, s):
        return ([0, 1, 0], 0)


class FakeGame:
    def __init__(self):
        self.num_terminal_calls = 0

    def initial_state(self):
        return [-1, -1, -1]

    def terminal(self, _):
        ''' Return true the second time this is called '''
        if self.num_terminal_calls > 0:
            return True
        self.num_terminal_calls += 1
        return False

    def next_state(self, _s, _a, _p):
        return [-1, 1, -1]

    def winner(self, _):
        return 0


def test_pit_networks():
    network1 = FakeNetwork()
    network2 = FakeNetwork()
    game = FakeGame()
    agent = AlphaZeroAgent(network1)
    percent_wins = agent.pit_networks(game, network1, network2, num_games=1)
    assert percent_wins == 0.0
