class Agent:
    '''
    An agent class which exposes a method called action. Given a certain
    state of a game and the player that is playing, the agent retuns the
    best action it can find, given a certain heuristic or strategy
    '''

    def action(self, g, s, p):
        '''
        Given a game, a state of the game, return an action
        '''
        raise NotImplementedError
