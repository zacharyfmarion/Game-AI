from gameai.algorithms import MCTS
from gameai.core import TrainableAgent


class MCTSAgent(TrainableAgent):
    '''
    Agent that uses Monte Carlo Tree Search (MCTS) 

    Attributes:
        mcts (MTCS): The mcts search class
    '''

    def __init__(self):
        self.mcts = MCTS()

    def train(self, g, **kwargs):
        self.mcts.search(g, **kwargs)

    def train_episode(self, g, **kwargs):
        self.mcts.search_episode(g, **kwargs)

    def training_params(self, _):
        return (self.mcts.plays, self.mcts.wins)

    def action(self, g, s, p):
        return self.mcts.best_action(g, s, p)
