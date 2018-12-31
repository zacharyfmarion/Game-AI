class Game:
    '''
    Game class, which is extended to implement different types of adversarial,
    zero sum games. The class itself is stateless and all methods are actually
    static.
    '''

    def initial_state(self):
        '''
        Return the initial state of the game
        '''
        raise NotImplementedError

    @property
    def total_action_space_size(self):
        '''
        Return the total size of the action space
        '''
        raise NotImplementedError

    def action_space(self, s):
        '''
        For any given state returns a list of all possible valid actions

        Args:
            s (any): The state of the game
        '''
        raise NotImplementedError

    def terminal(self, s):
        '''
        Returns whether a given state is terminal

        Args:
            s (any): The state of the game
        '''
        raise NotImplementedError

    def flip_state(self, s):
        '''
        Invert the state of the board so that player 0 becomes player 1

        Args:
            s (any): The state of the game
        '''
        raise NotImplementedError

    def winner(self, s):
        '''
        Returns the winner of a game, or -1 if there is no winner

        Args:
            s (any): The state of the game
        '''
        raise NotImplementedError

    def reward(self, s, p):
        '''
        Returns the reward for a given state

        Args:
            s (any): The state of the game
            p (int): The player to get the reward for
        '''
        raise NotImplementedError

    def next_state(self, s, a, p):
        '''
        Given a state, action, and player id, return the state resulting from the
        player making that move

        Args:
            s (any): The state of the game
            a (int): The action for the player to take
            p (int): The player to get the next state for
        '''
        raise NotImplementedError

    def to_readable_string(self, s):
        '''
        Returns a pretty-formatted representation of the board

        Args:
            s (any): The state of the game
        '''
        return str(s)

    def to_hash(self, s):
        '''
        Returns a hash of the game state, which is necessary for some algorithms
        such as MCTS

        Args:
            s (any): The state of the game
        '''
        return hash(str(s))
