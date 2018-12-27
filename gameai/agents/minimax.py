from core.agent import Agent


class MinimaxAgent(Agent):
    def __init__(self, **kwargs):
        Agent.__init__(self)
        # TODO: use this to terminate the traversal
        self.horizon = kwargs.get('horizon', float("inf"))

    def action(self, g, s, p):
        actions = g.action_space(s)
        rewards = [self.min_play(g, g.next_state(s, a, p), 1-p)
                   for a in actions]
        return actions[rewards.index(max(rewards))]

    # Helpers
    def min_play(self, g, s, p):
        actions = g.action_space(s)
        if g.terminal(s):
            return g.reward(s, 1-p)
        return min([self.max_play(g, g.next_state(s, a, p), 1-p) for a in actions])

    def max_play(self, g, s, p):
        actions = g.action_space(s)
        if g.terminal(s):
            return g.reward(s, p)
        return max([self.min_play(g, g.next_state(s, a, p), 1-p) for a in actions])
