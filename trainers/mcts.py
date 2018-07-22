import math
from tqdm import tqdm
from random import choice
from core.trainer import Trainer

class MCTSTrainer(Trainer):

  def __init__(self, **kwargs):
    self.wins = {}
    self.plays = {}
    self.C = kwargs.get('C', 1.4)

  def train(self, g, **kwargs):
    '''
    Play out a certain number of games, each time updating our win and play
    counts for any state that we visit during the game. As we continue to
    play, num_wins / num_plays for a given state should begin to converge on
    the true optimality of a state
    '''
    num_episodes = kwargs.get('num_episodes', 10000)
    verbose = kwargs.get('verbose', False)
    if verbose:
      for game in tqdm(range(num_episodes)):
        self.play_game(g)
    else:
      for game in range(num_episodes):
        self.play_game(g)
    return self.training_params()
  
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
      a = self.monte_carlo_action(g, s, p)
      s = g.next_state(s, a, p)
      p = 1 - p
      if g.terminal(s): break
    
    winner = g.winner(s)
    
    # Now we backpropagate the result of the game
    for p, s in visited:
      self.plays[(p, s)] = self.plays.get((p, s), 0) + 1
      self.wins[(p, s)] = self.wins.get((p, s), 0) + (1 if winner == p else 0)

  def monte_carlo_action(self, g, s, p):
    '''
    Choose an action during self play based on the UCB1 algorithm. Instead of just
    choosing the action that led to the most wins in the past, we choose the action
    that balances this concern with exploration
    '''
    actions = g.action_space(s)

    # Stop out early if there is only one choice
    if len(actions) == 1: return actions[0]

    next_state_hashes = [g.to_hash(g.next_state(s, a, p)) for a in actions]
    best_move = None

    # We first check that this player has been in each of the subsequent states
    # If they have not, then we simply choose a random action
    if all((1-p, s_hash) in self.plays for s_hash in next_state_hashes):

      log_total = math.log(sum(self.plays[(1-p, s_hash)] for s_hash in next_state_hashes))
      values = [
        (self.wins[(1-p, s_hash)] / self.plays[(1-p, s_hash)]) +
        self.C * math.sqrt(log_total / self.plays[(1-p, s_hash)])
        for s_hash in next_state_hashes
      ]

      # We want to minimize the q-value of our opponent, so we return the action
      # that yeilds the least amount of wins to the other player
      next_move_index = values.index(min(values))
      best_move = actions[next_move_index]
    else:
      best_move = choice(actions)

    return best_move
  
  def training_params(self):
    '''
    Return the params that we learned through self play
    '''
    return (self.plays, self.wins)