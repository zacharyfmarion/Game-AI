from random import choice
import sys

from core.agent import Agent
from games.tictactoe import TicTacToe

# Here's the idea: we want to learn how to play tic tac toe by keeping track 
# of the best action in any state. We will do this by propagating whether or 
# not the current player won the game back up through the game history. After
# enough iterations of game simulations we can choose the best move based on
# this stored information

class MCTSAgent(Agent):

  def __init__(self, **kwargs):
    self.wins = kwargs.get('plays', {})
    self.plays = kwargs.get('wins', {})
  
  def action(self, g, s, p):
    '''
    Given the state and player return the best action. Note that the player is
    the player that is about to play, so we use the opposite player when self.plays
    and self.wins for the next_state.
    '''
    actions = g.action_space(s)

    # Stop out early if there is only one choice
    if len(actions) == 1: return actions[0]

    next_state_hashes = [g.to_hash(g.next_state(s, a, p)) for a in actions]
    best_move = None

    # We first check that this player has been in each of the subsequent states
    # If they have not, then we simply choose a random action
    if all((p, s_hash) in self.plays for s_hash in next_state_hashes):

      q_values = [self.wins[(p, s_hash)] / self.plays[(p, s_hash)] \
                  for s_hash in next_state_hashes]

      # We want to minimize the q-value of our opponent, so we return the action
      # that yeilds the least amount of wins to the other player
      best_move_index = q_values.index(min(q_values))
      best_move = actions[best_move_index]
    else:
      best_move = choice(actions)

    return best_move