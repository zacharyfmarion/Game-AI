class Agent:
  def __init__(self, id):
    self.id = id

  def action(self, g, s):
    '''
    Given a game, a state of the game, return an action
    '''
    raise NotImplementedError

  def train(self):
    '''
    Train an agent. Only relevant for some agents
    '''
    print('Training not implmented for this agent')
  
  def __int__(self):
    '''
    Integer representation of the agent
    '''
    return self.id