from core.agent import Agent

class HumanAgent(Agent):
  def action(self, g, s, p):
    actions = g.action_space(s)
    print("Valid actions: {}".format(actions))
    action = int(input("move > "))
    if action in actions:
      return action
    return self.action(g, s, p)