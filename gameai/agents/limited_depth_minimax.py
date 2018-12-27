from core.agent import Agent


class LimitedDepthMinimaxAgent(Agent):

    def __init__(self, **kwargs):
        '''
        The horizon is how deep down the search tree that we go before evaluating
        our heuristic function
        '''
        self.horizon = kwargs.get('horizon', 4)

    def action(self, g, s, p):
        actions = g.action_space(s)
        rewards = [self.min_play(g, g.next_state(s, a, p), p, 0)
                   for a in actions]
        return actions[rewards.index(max(rewards))]

    # Helpers
    def min_play(self, g, s, p, depth):
        actions = g.action_space(s)
        if g.terminal(s) or depth >= self.horizon:
            return g.reward(s, p)
        return min([self.max_play(g, g.next_state(s, a, 1-p), 1-p, depth+1) for a in actions])

    def max_play(self, g, s, p, depth):
        actions = g.action_space(s)
        if g.terminal(s) or depth >= self.horizon:
            return g.reward(s, p)
        return max([self.min_play(g, g.next_state(s, a, p), p, depth+1) for a in actions])
