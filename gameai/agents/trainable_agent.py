from .agent import Agent


class TrainableAgent(Agent):
    '''
    Class that extends the functionality of a normal agent. This is necessary
    because agents are bound to a particular player, but for some algorithms
    the agent is really being trained to play optimally for both plays, so we
    have this class house the training data and then pass it into the agents
    when they are instantiated to avoid duplicated work
    '''

    def train(self, g, **kwargs):
        '''
        Train the agent. As a convenience this should return :code:`self.training_params()`
        at the end of training
        '''
        raise NotImplementedError

    def train_episode(self, g, **kwargs):
        '''
        Single training iteration
        '''
        raise NotImplementedError

    def training_params(self, g):
        '''
        Return the params that result from training
        '''
        raise NotImplementedError
