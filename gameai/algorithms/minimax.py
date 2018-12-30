from gameai.core import Algorithm


class Minimax(Algorithm):
    '''
    Implementation of the minimax algorithm.

    Attributes:
        horizon (int): The max depth of the search. Defaults to infinity. Note that if this
            is set then the game's hueristic is used
    '''

    def __init__(self, horizon=float('inf')):
        self.horizon = horizon

    def best_action(self, g, s, p):
        actions = g.action_space(s)
        rewards = [self.min_play(g, g.next_state(s, a, p), 1-p, 0)
                   for a in actions]
        return actions[rewards.index(max(rewards))]

    def min_play(self, g, s, p, depth):
        '''
        Get the smallest value of all the child nodes

        Args:
            g (Game): The game
            s (any): The state of the game upon execution
            p (int): The current player (who is about to make a move)
            depth (int): The current depth of the search tree

        Returns:
            int: The smallest value of all the child states
        '''
        actions = g.action_space(s)
        if g.terminal(s) or depth > self.horizon:
            return g.reward(s, 1-p)
        return min([self.max_play(g, g.next_state(s, a, p), 1-p, depth+1) for a in actions])

    def max_play(self, g, s, p, depth):
        '''
        Get the largest value of all the child nodes

        Args:
            g (Game): The game
            s (any): The state of the game upon execution
            p (int): The current player (who is about to make a move)
            depth (int): The current depth of the search tree

        Returns:
            int: The largest value of all the child states
        '''
        actions = g.action_space(s)
        if g.terminal(s) or depth > self.horizon:
            return g.reward(s, p)
        return max([self.min_play(g, g.next_state(s, a, p), 1-p, depth+1) for a in actions])
