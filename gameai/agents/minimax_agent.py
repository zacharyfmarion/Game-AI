from gameai.algorithms import Minimax
from gameai.core import Agent


class MinimaxAgent(Agent):
    '''
    Implementation of minimax which allows you to specify a cutoff horizon for the
    search.

    Attributes:
        minimax (Minimax): Algorithm that runs the minimax search
    '''

    def __init__(self, **kwargs):
        '''
        The horizon is how deep down the search tree that we go before evaluating
        our heuristic function
        '''
        self.minimax = Minimax(**kwargs)

    def action(self, g, s, p):
        return self.minimax.best_action(g, s, p)
