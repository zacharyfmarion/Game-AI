from random import choice

from .player import Player


class Arena:
    '''
    Place where two agents are pitted against eachother in a series of games.
    Statistics on the win rates are recorded and can be displayed.

    Attributes:
        game (Game): The game that is being played
        players (list): List of Player objects. Note that there should only be two, and
            the ids of the player should map to the index of the player in the array.
        games_played (int): The number of games played in the arena
        wins (list): List of two integers representing the number of wins of each player,
            with the index being the id of the player
    '''

    def __init__(self, game, players):
        '''
        Note:
            The `players` argument is a list of players to be used. In the future, when
            more than two players are supported this can be generalized to n players.

        Args:
            game (Game)
            players (list)
        '''
        if not all(isinstance(p, Player) for p in players):
            raise ValueError('Expected `model` argument to be a list of '
                             '`Player` instances, got ', players)
        if len(players) != 2:
            raise ValueError('There should be two players in every game')
        self.game = game
        self.players = players
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
            winner = self.play_game(verbose)
            if winner in [p.player_id for p in self.players]:
                self.wins[winner] += 1

    def statistics(self):
        '''
        Print out the statistics for a given series of games.
        '''
        if self.games_played == 0:
            raise ZeroDivisionError(
                'At least one game must be played before statistics can be generated')
        for i in range(len(self.players)):
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
            We always have the start with player being 0 from the persepctive
            of the agent. Because of this we pass in a :code:`flip` boolean to
            the player class in the action method, which flips the board and
            makes it seems as though player 0 started, even if it was actually
            player 1

        Args:
            verbose (bool): Whether or not to print the output of the game.
                Defaults to false

        Returns:
            int: The winner of the game
        '''
        state = self.game.initial_state()
        starting_player = choice([p.player_id for p in self.players])
        player = starting_player

        # Play out the full game
        while not self.game.terminal(state):
            action = self.players[player].action(
                self.game, state, starting_player != 0)
            state[action] = player
            if verbose:
                print(self.game.to_readable_string(state), "\n")
            player = 1 - player

        self.games_played += 1
        return self.game.winner(state)
