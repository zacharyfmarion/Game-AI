import math
from gameai.core import Game


class TicTacToe(Game):
    '''
    Implements a 3x3 game of tictactoe, with state represented as an array of length 9.
    Currently the implementation is somewhat brittle and cannot be extended to an nxn
    board easily.

    Examples:
        >>> TicTacToe().initial_state()
        [-1, -1, -1, -1, -1, -1, -1, -1, -1]

        >>> TicTacToe().to_readable_string([-1, 1, -1, 0, 0, -1, -1, 1, -1])
           | O |
        -----------
         X | X |
        -----------
           | O |
    '''

    def initial_state(self):
        return [-1 for i in range(9)]

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

    def to_readable_string(self, s):
        board = ""
        for i, player in enumerate(s):
            end_of_line = (i + 1) % math.sqrt(len(s)) != 0
            row_line = "\n-----------\n" if i != len(s) - 1 else ""
            board += " {} {}".format(self.stringify_player(player),
                                     "|" if end_of_line else row_line)
        return board

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
        be generalized to an nxn board.
        '''
        return ((s[6] == p and s[7] == p and s[8] == p) or
                (s[3] == p and s[4] == p and s[5] == p) or
                (s[0] == p and s[1] == p and s[2] == p) or
                (s[6] == p and s[3] == p and s[0] == p) or
                (s[7] == p and s[4] == p and s[1] == p) or
                (s[8] == p and s[5] == p and s[2] == p) or
                (s[6] == p and s[4] == p and s[2] == p) or
                (s[8] == p and s[4] == p and s[0] == p))

    @staticmethod
    def stringify_player(tile):
        mapping = dict(enumerate(['X', 'O']))
        return mapping.get(tile, ' ')
