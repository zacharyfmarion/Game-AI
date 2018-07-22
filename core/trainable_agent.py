class TrainableAgent:
  def action(self, g, s, p):
    '''
    Given a game, a state of the game, and player, return an action. Note that
    in contrast to the Agent class, we do not assign an id. This is because the
    agent learns the optimal play for both players, and so can simply play itself
    intead of playing a distinct opponent.
    '''
    raise NotImplementedError

  def train(self, g):
    '''
    Train an agent. Only relevant for some agents
    '''
    raise NotImplementedError