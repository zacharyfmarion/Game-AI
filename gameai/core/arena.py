from random import choice

from .player import Player


class Arena:
    '''
    Place where two agents are pitted against eachother in a series of games.
    Statistics on the win rates are recorded and can be displayed.
    '''

    def __init__(self, game, players):
        '''
        The `players` argument is a list of players to be used. In the future, when
        more than two players are supported this can be generalized to n players.
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

        NOTE: We always have the start with player being 0 from the persepctive
        of the agent. Because of this we pass in a 'flip' boolean to the player
        class in the action method, which flips the board and makes it seems as
        though player 0 started, even if it was actually player 1
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
