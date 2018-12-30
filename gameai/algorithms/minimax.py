class Minimax:
    '''
    Implementation of the minimax algorithm.

    Attributes:
        horizon (int): The max depth of the search. Defaults to infinity. Note that if this
            is set then the game's hueristic is used
    '''

    def __init__(self, **kwargs):
        self.horizon = kwargs.get('horizon', float('inf'))

    def best_action(self, g, s, p):
        actions = g.action_space(s)
        rewards = [self.min_play(g, g.next_state(s, a, p), 1-p, 0)
                   for a in actions]
        return actions[rewards.index(max(rewards))]

    def min_play(self, g, s, p, depth):
        '''
        Return the smallest value out of all of the child states
        '''
        actions = g.action_space(s)
        if g.terminal(s) or depth > self.horizon:
            return g.reward(s, 1-p)
        return min([self.max_play(g, g.next_state(s, a, p), 1-p, depth+1) for a in actions])

    def max_play(self, g, s, p, depth):
        '''
        Return the largest value out of all of the child states
        '''
        actions = g.action_space(s)
        if g.terminal(s) or depth > self.horizon:
            return g.reward(s, p)
        return max([self.min_play(g, g.next_state(s, a, p), 1-p, depth+1) for a in actions])
