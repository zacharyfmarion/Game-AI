class Agent:
    '''
    An agent class which exposes a method called action. Given a certain
    state of a game and the player that is playing, the agent retuns the
    best action it can find, given a certain heuristic or strategy
    '''

    def action(self, g, s, p):
        '''
        Given a game, a state of the game, return an action

        Args:
            g (Game): The game the agent is competing in
            s (any): The state of the game
            p (int): The current player (either 0 or 1)

        Returns:
            int: The index of the action within the returned action space
        '''
        raise NotImplementedError
