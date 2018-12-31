from copy import deepcopy
from random import shuffle
from tqdm import tqdm
import numpy as np

from gameai.core import TrainableAgent
from gameai.algorithms import MCTS


class AlphaZeroAgent(TrainableAgent):
    '''
    An alphazero agent

    Attributes:
        nnet (Network): The backing neural network for the agent
    '''

    def __init__(self, nnet):
        self.nnet = nnet

    def train(self, g, **kwargs):
        '''
        Train the agent for a certain number of iterations
        '''
        num_iters = kwargs.get('num_iters', 100)
        verbose = kwargs.get('verbose', False)
        num_episodes = kwargs.get('num_episodes', 100)
        win_threshold = kwargs.get('win_threshold', .55)
#
        examples = []
        iter_wrapper = tqdm if verbose else lambda x: x

        for _ in iter_wrapper(range(num_iters)):
            for _ in range(num_episodes):
                examples += self.train_episode(g, **kwargs)

            new_nnet = self.copy_and_train(self.nnet, examples)
            win_percentage = self.pit_networks(g, new_nnet, self.nnet)
            if win_percentage >= win_threshold:
                self.nnet = new_nnet

        return self.nnet

    def train_episode(self, g, **kwargs):
        '''
        Train a single episode of the network
        '''
        num_simulations = kwargs.get('num_simulations', 100)
        p = 0
        s = g.initial_state()
        mcts = MCTS()
        examples = []
        while True:
            mcts.search(g, s, p, num_iters=num_simulations, nnet=self.nnet)
            policy = mcts.policy(g, s)
            examples.append([s, policy, None])
            a = np.random.choice(len(policy), p=policy)
            s = g.next_state(s, a, p)
            p = 1 - p

            if g.terminal(s):
                examples = self.assign_rewards(examples, g.winner(s))
                return examples

    def training_params(self, g):
        return self.nnet

    def action(self, _g, s, _p):
        return self.nnet.predict_single(s)

    @staticmethod
    def copy_and_train(nnet, examples):
        '''
        Return a copy of the passed in network, and train that copy
        on the examples given

        Args:
            nnet (Network): The network to copy
            examples (list): List of examples of the form :code:`[state, policies, reward]`

        Returns:
            Network: The network copy
        '''
        new_nnet = deepcopy(nnet)
        new_nnet.train(examples)
        return new_nnet

    @staticmethod
    def pit_networks(g, new_nnet, old_nnet, num_games=50):
        '''
        Pit two networks against eachother in a game

        Args:
            g (Game): The game the networks are playing
            new_nnet (Network): The network trained on the latest examples
            old_nnet (Network): The previous best network
            num_games (int): The number of games to play

        Returns:
            float: The win percentage of the new network
        '''
        num_wins = 0
        for _ in range(num_games):
            s = g.initial_state()
            nets = [new_nnet, old_nnet]
            shuffle(nets)

            p = 0
            while not g.terminal(s):
                a, _ = nets[p].predict_single(s)
                s = g.next_state(s, a, p)
                p = 1 - p

            if nets.index(new_nnet) == g.winner(s):
                num_wins += 1

        return num_wins / float(num_games)

    @staticmethod
    def assign_rewards(examples, winner):
        '''
        TODO
        '''
        return [[s, policy, 1 if winner == 0 else 0] for [s, policy, _] in examples]
