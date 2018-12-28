import math
from random import choice
from tqdm import tqdm

from .trainer import Trainer
from .utils import assign_rewards


class MCTSTrainer(Trainer):

    def __init__(self, **kwargs):
        self.wins = {}
        self.plays = {}
        self.num_iters = kwargs.get('num_iters', 100)
        self.num_episodes = kwargs.get('num_episodes', 100)
        self.verbose = kwargs.get('verbose', False)
        self.c_punt = kwargs.get('c_punt', 1.4)

    def train(self, g, **kwargs):
        '''
        Play out a certain number of games, each time updating our win and play
        counts for any state that we visit during the game. As we continue to
        play, num_wins / num_plays for a given state should begin to converge on
        the true optimality of a state
        '''
        if self.verbose:
            for _ in tqdm(range(self.num_iters)):
                self.train_episodes(g)
        else:
            for _ in range(self.num_iters):
                self.train_episodes(g)
        return self.training_params(g)

    def train_episodes(self, g):
        examples = []
        for _ in range(self.num_episodes):
            examples += self.train_episode(g)
        self.update(examples)

    def train_episode(self, g):
        '''
        We play a game by starting in the boards starting state and then
        choosing a random move. We then move to the next state, keeping
        track of which moves we chose. At the end of the game we go through
        our visited list and update the values of wins and plays so that we
        have a better understanding of which states are good and which are bad
        '''
        s = g.initial_state()
        p = 0
        examples = []
        while True:
            # Update visited with the next state
            a = self.monte_carlo_action(g, s, p)
            s = g.next_state(s, a, p)
            examples.append([p, g.to_hash(s), None])
            p = 1 - p
            if g.terminal(s):
                examples = assign_rewards(examples, g.winner(s))
                return examples

    def monte_carlo_action(self, g, s, p):
        '''
        Choose an action during self play based on the UCB1 algorithm. Instead of just
        choosing the action that led to the most wins in the past, we choose the action
        that balances this concern with exploration
        '''
        actions = g.action_space(s)

        # Stop out early if there is only one choice
        if len(actions) == 1:
            return actions[0]

        next_state_hashes = [g.to_hash(g.next_state(s, a, p)) for a in actions]
        best_move = None

        # We first check that this player has been in each of the subsequent states
        # If they have not, then we simply choose a random action
        if all((p, s_hash) in self.plays for s_hash in next_state_hashes):

            log_total = math.log(
                sum(self.plays[(p, s_hash)] for s_hash in next_state_hashes))
            values = [
                (self.wins[(p, s_hash)] / self.plays[(p, s_hash)]) +
                self.c_punt * math.sqrt(log_total / self.plays[(p, s_hash)])
                for s_hash in next_state_hashes
            ]

            next_move_index = values.index(max(values))
            best_move = actions[next_move_index]
        else:
            best_move = choice(actions)

        return best_move

    def update(self, examples):
        '''
        Backpropagate the result of the training episodes
        '''
        for [p, s, reward] in examples:
            self.plays[(p, s)] = self.plays.get((p, s), 0) + 1
            self.wins[(p, s)] = self.wins.get((p, s), 0) + reward

    def training_params(self, _):
        '''
        Return the params that we learned through self play
        '''
        return (self.plays, self.wins)
