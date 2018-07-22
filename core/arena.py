class Arena:
  '''
  Place where two agents are pitted against eachother in a series of games.
  Statistics on the win rates are recorded and can be displayed.
  '''

  def __init__(self, game, agents):
    '''
    agents is a list of agents to be used. In the future, when more than two 
    players are supported this can be generalized to n players.
    '''
    if len(agents) != 2: raise ValueError
    self.game = game
    self.agents = agents
    self.games_played = 0
    self.wins = [0, 0]

  def play_games(self, num_games = 10):
    for game in range(num_games):
      winner = self.play_game() 
      if winner in [0, 1]: 
        self.wins[winner] += 1

  def statistics(self):
    for i in range(2):
      print('Player {}: \n  - Games: {} / {}\n  - Percentage: {}%'.format(
        i, 
        self.wins[i], 
        self.games_played, 
        (self.wins[i] / self.games_played) * 100
      ))

  def play_game(self, verbose=False):
    state = self.game.initial_state()
    player = 0

    # Play out the full game
    while not self.game.terminal(state):
      action = self.agents[player].action(self.game, state, player)
      state[action] = player
      if verbose:
        print(self.game.to_readable_string(state), "\n")
      player = 1 - player

    self.games_played += 1
    return self.game.winner(state)