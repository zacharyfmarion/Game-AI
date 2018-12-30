from gameai.core import Game


class LineTacToe(Game):
    '''
    Implements a 1x3 tictactoe-like, with state represented as an array of length 3.
    The goal of the game is to get two consecutive xs or os. For example, [o, o, x]
    is winning for o. Note that whoever starts the game should win, every time, as
    going in the center will win the game. However this is a good game to test new
    agent / algorithm implementations as the entire state space is only 11 states.

    Examples:
        >>> LineTacToe().initial_state()
        [-1, -1, -1]
    '''

    def initial_state(self):
        return [-1 for i in range(3)]

    def action_space(self, s):
        return [i for i in range(len(s)) if s[i] == -1]

    def terminal(self, s):
        return self.is_winner(s, 0) or self.is_winner(s, 1) or len(self.action_space(s)) == 0

    def flip_state(self, s):
        def state_map(p):
            if not p in [0, 1]:
                return p
            return 1 - p
        return [state_map(p) for p in s]

    def winner(self, s):
        if self.is_winner(s, 0):
            return 0
        if self.is_winner(s, 1):
            return 1
        return -1

    def reward(self, s, p):
        if self.is_winner(s, p):
            return 1
        if self.is_winner(s, 1-p):
            return -1
        return self.heuristic(s)

    def next_state(self, s, a, p):
        copy = s.copy()
        copy[a] = p
        return copy

    def to_hash(self, s):
        return hash(tuple(s))

    @staticmethod
    def heuristic(_):
        ''' Stubbed for now '''
        return 0

    @staticmethod
    def is_winner(s, p):
        '''
        Return whether a particular player has won the game. Ideally this would
        be generalized to a 1xn board.
        '''
        return ((s[0] == p and s[1] == p) or
                (s[1] == p and s[2] == p))
