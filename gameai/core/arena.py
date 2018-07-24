from core.player import Player

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

  def play_games(self, num_games = 10):
    '''
    Play a series of games between the players, recording how they did
    so that we can display statistics on which player performed better
    '''
    for game in range(num_games):
      winner = self.play_game() 
      if winner in [p.id for p in self.players]: 
        self.wins[winner] += 1

  def statistics(self):
    '''
    Print out the statistics for a given series of games.
    '''
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
    accurate statistics and returning the winner (or -1 if no winner)
    '''
    state = self.game.initial_state()
    player = self.players[0].id

    # Play out the full game
    while not self.game.terminal(state):
      action = self.players[player].action(self.game, state)
      state[action] = player
      if verbose:
        print(self.game.to_readable_string(state), "\n")
      # TODO: Make this generalized for more than 2 players
      player = 1 - player

    self.games_played += 1
    return self.game.winner(state)