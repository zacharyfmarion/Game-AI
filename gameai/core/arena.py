from random import shuffle

from .agent import Agent


class Arena:
    '''
    Place where two agents are pitted against eachother in a series of games.
    Statistics on the win rates are recorded and can be displayed.

    Attributes:
        game (Game): The game that is being played
        agents (list): List of Agent objects. Note that there should only be two
        games_played (int): The number of games played in the arena
        wins (list): List of two integers representing the number of wins of each player,
            with the index being the id of the player
    '''

    def __init__(self, game, agents):
        if not all(isinstance(agent, Agent) for agent in agents):
            raise ValueError('Expected `model` argument to be a list of '
                             '`Agent` instances, got ', agents)
        if len(agents) != 2:
            raise ValueError('There should be two players in every game')
        self.game = game
        self.agents = agents
        self.games_played = 0
        self.wins = [0, 0]

    def play_games(self, **kwargs):
        '''
        Play a series of games between the players, recording how they did
        so that we can display statistics on which player performed better

        Args:
            num_episodes (int): The number of games to play, defaults to 10
            verbose (bool): Whether or not to print output from each game.
                Defaults to false
        '''
        num_episodes = kwargs.get('num_episodes', 10)
        verbose = kwargs.get('verbose', False)

        for _ in range(num_episodes):
            self.play_game(verbose)

    def statistics(self):
        '''
        Print out the statistics for a given series of games.
        '''
        if self.games_played == 0:
            raise ZeroDivisionError(
                'At least one game must be played before statistics can be generated')
        for i in range(len(self.agents)):
            print('Player {}: \n  - Games: {} / {}\n  - Percentage: {}%'.format(
                i,
                self.wins[i],
                self.games_played,
                (self.wins[i] / self.games_played) * 100
            ))

    def play_game(self, verbose=False):
        '''
        Play a single game, doing the necessary bookkeeping to maintain
        accurate statistics and returning the winner (or -1 if no winner).

        Note:
            We always have the start with player being 0 from the persective
            of the agent. Because of this we flips the board for the agent going
            second and make it seems as though the agent started

        Args:
            verbose (bool): Whether or not to print the output of the game.
                Defaults to false

        Returns:
            int: The winner of the game
        '''
        agents = self.agents.copy()
        shuffle(agents)

        p = 0
        g = self.game
        s = g.initial_state()

        if verbose:
            print("Agent {} going first".format(self.agents.index(agents[0])))

        # Play out the full game
        while not g.terminal(s):
            a = agents[p].action(g, s, p)
            s = g.next_state(s, a, p)
            p = 1 - p
            if verbose:
                print(g.to_readable_string(s), "\n")

        self.games_played += 1
        winner = g.winner(s)

        if winner in [0, 1]:
            winning_agent = self.agents.index(agents[winner])
            self.wins[winning_agent] += 1

        return winner
