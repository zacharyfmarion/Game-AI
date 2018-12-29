class Player:
    '''
    Player is a simple abstraction whose policy is defined by the
    agent that backs it. Agents learn the optimal play for each player,
    while players are only concerned about the optimal play for
    themselves

    Attributes:
        player_id (int): The id of the player
        agent (Agent): The agent associated with the player

    Raises:
        ValueError: If the id is not 0 or 1
    '''

    def __init__(self, player_id, agent):
        if player_id not in [0, 1]:
            raise ValueError('Expected an id of 0 or 1')
        self.player_id = player_id
        self.agent = agent

    def action(self, g, s, flip):
        '''
        Take an action with the backing agent. If the starting player is
        not 0, then we invert the board so that the starting player is still
        0 from the perspective of the agent

        Args:
            g (Game): The game the player is playing
            s (any): The state of the game
            flip (bool): Whether or not to flip the state so that the agent
                thinks that player 0 started the game. This is necessary since
                trainable agents like MCTSAgent operate under the assumption that
                player 0 always starts

        Returns:
            int: The index of the action the player will take
        '''
        state = g.flip_state(s) if flip else s
        player = 1 - self.player_id if flip else self.player_id
        return self.agent.action(g, state, player)
