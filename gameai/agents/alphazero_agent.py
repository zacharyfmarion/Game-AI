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
        network_generator (lambda): function with the signiature :code:`lambda weights: (method body)`
            that generates a network to be used in the agent
    '''

    def __init__(self, network_generator):
        self.nnet = network_generator()
        self.network_generator = network_generator

    def train(self, g, **kwargs):
        '''
        Train the agent for a certain number of iterations
        '''
        num_iters = kwargs.get('num_iters', 100)
        verbose = kwargs.get('verbose', False)
        num_episodes = kwargs.get('num_episodes', 10)
        win_threshold = kwargs.get('win_threshold', .55)
#
        examples = []
        iter_wrapper = tqdm if verbose else lambda x: x

        for _ in iter_wrapper(range(num_iters)):
            for episode in range(num_episodes):
                if verbose:
                    print('EPISODE: {}'.format(episode))
                examples += self.train_episode(g, **kwargs)

            new_nnet = self.copy_and_train(self.nnet, examples)
            win_percentage = self.pit_networks(
                g, new_nnet, self.nnet, verbose=verbose)
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
        while not g.terminal(s):
            mcts.search(g, s, p, num_iters=num_simulations, nnet=self.nnet)
            policy = mcts.policy(g, s)
            examples.append([s, policy, None])
            a = np.random.choice(len(policy), p=policy)
            s = g.next_state(s, a, p)
            p = 1 - p

        examples = self.assign_rewards(examples, g.reward(s, p))
        return examples

    def training_params(self, g):
        return self.nnet

    def action(self, g, s, _p):
        policy, _ = self.nnet.predict_single(s)
        return self.get_best_valid_action(g, s, policy)

    def copy_and_train(self, nnet, examples):
        '''
        Return a copy of the passed in network, and train that copy
        on the examples given

        Args:
            nnet (Network): The network to copy
            examples (list): List of examples of the form :code:`[state, policies, reward]`

        Returns:
            Network: The network copy
        '''
        weights = nnet.weights()
        new_nnet = self.network_generator(weights)
        new_nnet.train(examples)
        return new_nnet

    def pit_networks(self, g, new_nnet, old_nnet, num_games=50, verbose=False):
        '''
        Pit two networks against eachother in a game

        Args:
            g (Game): The game the networks are playing
            new_nnet (Network): The network trained on the latest examples
            old_nnet (Network): The previous best network
            num_games (int): The number of games to play
            verbose (bool): Whether or not to print output of game progress

        Returns:
            float: The win percentage of the new network
        '''
        num_wins = 0
        iter_wrapper = tqdm if verbose else lambda x: x
        if verbose:
            print('Pitting networks:\n')
        for _ in iter_wrapper(range(num_games)):
            s = g.initial_state()
            nets = [new_nnet, old_nnet]
            shuffle(nets)

            p = 0
            while not g.terminal(s):
                policy, _ = nets[p].predict_single(s)
                a = self.get_best_valid_action(g, s, policy)
                s = g.next_state(s, a, p)
                p = 1 - p

            if nets.index(new_nnet) == g.winner(s):
                num_wins += 1

        win_percentage = num_wins / float(num_games)
        if verbose:
            print('New net win percentage: {}%'.format(
                win_percentage*100))

        return win_percentage

    @staticmethod
    def get_best_valid_action(g, s, policy):
        '''
        Given a state and a policy returned from the network, return the
        best valid action

        Args:
            g (Game): The game
            s (list): The current state
            poliy (list): The policy returned from the network

        Returns:
            int: The best valid action
        '''
        valid_actions = g.action_space(s)
        valid_policy = [
            policy[i] if i in valid_actions else 0 for i in range(len(policy))]
        return np.argmax(valid_policy)

    @staticmethod
    def assign_rewards(examples, reward):
        '''
        TODO
        '''
        return [[s, policy, reward] for [s, policy, _] in examples]
