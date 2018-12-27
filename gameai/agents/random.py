from random import choice

from .agent import Agent


class RandomAgent(Agent):
    def action(self, g, s, _):
        return choice(g.action_space(s))
