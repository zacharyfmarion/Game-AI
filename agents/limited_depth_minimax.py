from core.agent import Agent

class LimitedDepthMinimaxAgent(Agent):

  def __init__(self, id, **kwargs):
    '''
    The horizon is how deep down the search tree that we go before evaluating 
    our heuristic function
    '''
    Agent.__init__(self, id)
    self.horizon = kwargs.get('h', 4)

  def action(self, g, s):
    actions = g.action_space(s)
    rewards = [self.min_play(g, g.next_state(s, a, self.id), 0) for a in actions]
    return actions[rewards.index(max(rewards))]

  # Helpers
  def min_play(self, g, s, d):
    actions = g.action_space(s)
    if g.terminal(s) or d == self.horizon: return g.reward(s, self.id)
    return min([self.max_play(g, g.next_state(s, a, 1-self.id), d+1) for a in actions])

  def max_play(self, g, s, d):
    actions = g.action_space(s)
    if g.terminal(s) or d == self.horizon: return g.reward(s, self.id)
    return max([self.min_play(g, g.next_state(s, a, self.id), d+1) for a in actions])