from .agent import Agent


class HumanAgent(Agent):
    '''
    Human agent, which waits for human input to determine what action to take. Note that they
    should input an integer corresponding to the index of the action they want to select
    '''

    def action(self, g, s, p):
        actions = g.action_space(s)
        print("Valid actions: {}".format(actions))
        action = int(input("move > "))
        if action in actions:
            return action
        return self.action(g, s, p)
