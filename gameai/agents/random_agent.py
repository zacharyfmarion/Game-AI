from random import choice

from .agent import Agent


class RandomAgent(Agent):
    '''
    Implementation of a random agent, which simply selects a random action
    from the current action space each turn
    '''

    def action(self, g, s, _):
        return choice(g.action_space(s))
