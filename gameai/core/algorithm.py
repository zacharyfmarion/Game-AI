class Algorithm:
    '''
    A basic abstraction for a class that finds an action to take in
    a given state for a given player. Even if the algorithm is not stateful
    it is still implemented as a class to provide a uniform interface.

    Note:
        Despite this interface being almost identical to an agent, agents
        can use multiple algorithms to come up with an action for a player to
        execute in a game.
    '''

    def best_action(self, g, s, p):
        '''
        Return the best action given a state and player

        Args:
            g (Game): The game object
            s (any): The current state of the game
            p (int): The current player

        Returns:
            int: The best action the algorithm can find
        '''
        raise NotImplementedError
