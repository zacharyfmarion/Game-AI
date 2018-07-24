from random import choice
from core.agent import Agent

class RandomAgent(Agent):
  def action(self, g, s, p):
    return choice(g.action_space(s))