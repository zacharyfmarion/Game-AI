from random import choice

from .agent import Agent


class MCTSAgent(Agent):
    '''
    Here's the idea: we want to learn how to play a game by keeping track
    of the best action in any state. We will do this by propagating whether or
    not the current player won the game back up through the game history. After
    enough iterations of game simulations we can choose the best move based on
    this stored information
    '''

    def __init__(self, **kwargs):
        self.plays = kwargs.get('plays', {})
        self.wins = kwargs.get('wins', {})

    def action(self, g, s, p):
        '''
        Given the state and player return the best action. 
        '''
        actions = g.action_space(s)

        # Stop out early if there is only one choice
        if len(actions) == 1:
            return actions[0]

        best_move = None
        next_state_hashes = [
            g.to_hash(g.next_state(s, a, p)) for a in actions]

        # We first check that this player has been in each of the subsequent states
        # If they have not, then we simply choose a random action
        if all((p, s_hash) in self.plays for s_hash in next_state_hashes):
            q_values = [self.wins[(p, s_hash)] / self.plays[(p, s_hash)]
                        for s_hash in next_state_hashes]
            best_move_index = q_values.index(max(q_values))
            best_move = actions[best_move_index]
        else:
            best_move = choice(actions)

        return best_move
