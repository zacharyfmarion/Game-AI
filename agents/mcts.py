import numpy as np
import random
import sys

from core.agent import Agent
from games.tictactoe import TicTacToe

# Here's the idea: I want to learn how to play tic tac toe
# by keeping track of the best action in any state. I will
# do this by propagating whether or not I won the game back
# up through the game history

class MCTSAgent(Agent):
  
  def __init__(self, id, **kwargs):
    Agent.__init__(self, id)
    self.wins = {}
    self.plays = {}
  
  def action(self, g, s, p):
    '''
    Given the state and player return the best action. Note that the player is
    the player that is about to play, so we use the opposite player when self.plays
    and self.wins for the next_state
    '''
    actions = g.action_space(s)
    next_state_hashes = [g.to_hash(g.next_state(s, a, p)) for a in actions]
    best_move = None

    # We first check that this player has been in each of the subsequent states
    # If they have not, then we simply choose a random action

    if all((1-p, s_hash) in self.plays for s_hash in next_state_hashes):
      q_values = [self.wins[(1-p, s_hash)] / self.plays[(1-p, s_hash)] \
                  for s_hash in next_state_hashes]
      print(q_values)
      # We want to minimize the q-value of our opponent, so we return the action
      # that yeilds the least amount of wins to the other player
      best_move_index = q_values.index(min(q_values))
      best_move = actions[best_move_index]
    else:
      best_move = random.choice(actions)

    return best_move
  
  def train(self, g, num_games=100000):
    '''
    Play out a certain number of games, each time updating our win and play
    counts for any state that we visit during the game. As we continue to
    play, num_wins / num_plays for a given state should begin to converge on
    the true optimality of a state
    '''
    for game in range(num_games):
      self.play_game(g)
  
  def evaluate(self):
    for p, s in self.plays.keys():
      print("{}, {}: {}".format(p, s, self.wins[(p, s)] / self.plays[(p, s)]))
  
  def play_game_with_policy(self, g):
    '''
    After some simulations we attempt to play a game using our policy, 
    assuming that all actions have been visited from a given state
    '''
    p = 0
    s = g.initial_state()
    while True:
      a = self.action(g, s, p)
      s = g.next_state(s, a, p)
      print(g.to_readable_string(s), '\n')
      if g.terminal(s): break
      p = 1 - p
  
  def play_game(self, g):
    '''
    We play a game by starting in the boards starting state and then 
    choosing a random move. We then move to the next state, keeping 
    track of which moves we chose. At the end of the game we go through 
    our visited list and update the values of wins and plays so that we 
    have a better understanding of which states are good and which are bad
    '''
    s = g.initial_state()
    p = 0
    visited = []
    while True:
      # Update visited with the next state
      visited.append((p, g.to_hash(s)))
      actions = g.action_space(s)
      action = random.choice(actions)
      next_state = g.next_state(s, action, p)
      s = next_state
      p = 1 - p
      if g.terminal(s): break
    
    winner = g.winner(s)
    
    # Now we backpropagate the result of the game
    for p, s in visited:
      self.plays[(p, s)] = self.plays.get((p, s), 0) + 1
      self.wins[(p, s)] = self.wins.get((p, s), 0) + (1 if winner == p else 0)