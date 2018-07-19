class Agent:
  def action(self, g, s, p):
    '''
    Given a game, a state of the game, return an action
    '''
    raise NotImplementedError

  def train(self):
    '''
    Train an agent. Only relevant for some agents
    '''
    print('Training not implemnted for this agent')