'''
Player is a simple abstraction whose policy is defined by the
agent that backs it. Agents learn the optimal play for each player,
while players are only concerned about the optimal play for 
themselves
'''
class Player:
  def __init__(self, pid, agent):
    self.id = pid
    self.agent = agent
  
  def action(self, g, s):
    '''
    Note that we pass in our id as the player field here, which is really
    the entire purpose of this class. 
    '''
    return self.agent.action(g, s, self.id)