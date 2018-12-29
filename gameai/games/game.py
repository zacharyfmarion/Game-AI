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

    def action_space(self, s):
        '''
        For any given state returns a list of all possible valid actions
        '''
        raise NotImplementedError

    def terminal(self, s):
        '''
        Returns whether a given state is terminal
        '''
        raise NotImplementedError

    def flip_state(self, s):
        '''
        Invert the state of the board so that player 0 becomes player 1
        '''
        raise NotImplementedError

    def winner(self, s):
        '''
        Returns the winner of a game, or -1 if there is no winner
        '''
        raise NotImplementedError

    def reward(self, s, p):
        '''
        Returns the reward for a given state
        '''
        raise NotImplementedError

    def next_state(self, s, a, p):
        '''
        Given a state, action, and player id, return the state resulting from the
        player making that move
        '''
        raise NotImplementedError

    def to_readable_string(self, s):
        '''
        Returns a pretty-formatted representation of the board
        '''
        return str(s)

    def to_hash(self, s):
        '''
        Returns a hash of the game state, which is necessary for some algorithms
        such as MCTS
        '''
        return hash(str(s))
