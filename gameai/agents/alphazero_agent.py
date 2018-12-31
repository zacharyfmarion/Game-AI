from copy import deepcopy
from tqdm import tqdm

from gameai.core import TrainableAgent


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
        print('train episode')

    def training_params(self, g):
        return self.nnet

    def action(self, g, s, p):
        pass

    @staticmethod
    def copy_and_train(nnet, examples):
        '''
        Return a copy of the passed in network, and train that copy
        on the examples given

        Args:
            nnet (Network): The network to copy
            examples (list): List of examples of the form TODO
        '''
        new_nnet = deepcopy(nnet)
        new_nnet.train(examples)
        return new_nnet

    @staticmethod
    def pit_networks(g, new_nnet, old_nnet):
        '''
        Pit two networks against eachother in a game

        Args:
            g (Game): The game the networks are playing
            new_nnet (Network): The network trained on the latest examples
            old_nnet (Network): The previous best network

        Returns:
            float: The win percentage of the new network
        '''
        print("PITTING NETWORKS")
        return 0
