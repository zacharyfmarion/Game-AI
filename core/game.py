class Game:
  def initial_state(self):
    '''
    Return the initial state of the game
    '''
    raise NotImplementedError

  def action_space(self, s):
    '''
    For any given state returns a list of all possible valid actions
    '''
    raise NotImplementedError

  def terminal(self, s):
    '''
    Returns whether a given state is terminal
    '''
    raise NotImplementedError

  def reward(self, s, p):
    '''
    Returns the reward for a given state 
    '''
    raise NotImplementedError
  
  def next_state(self, s, a, p):
    '''
    Given a state, action, and player id, return the state resulting from the 
    player making that move
    '''
    raise NotImplementedError

  def to_string(self, s):
    '''
    Returns a string representation of the board
    '''
    return str(s)