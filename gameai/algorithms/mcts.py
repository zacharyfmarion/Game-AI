import math
from random import choice
from tqdm import tqdm

from .utils import assign_rewards

DEFAULT_C_PUNT = 1.4


class MCTS:
    '''
    Implementation of a Monte Carlo Tree Search. We want to learn how to play
    a game by keeping track of the best action in any state. We will do this
    by propagating whether or not the current player won the game back up through
    the game history. After enough iterations of game simulations we can choose
    the best move based on this stored information

    Attributes:
        wins (dict): A dictionary where the key is a tuple :code:`(player, state_hash)`
            and the value is the number of wins that occurred at that state for the
            player. Note that the player represents whoever *played* the move in the state.
        plays (dict): A dictionary of the same format as wins which represents the
            number of times the player made a move in the given state
    '''

    def __init__(self):
        self.wins = {}
        self.plays = {}

    def search(self, g, num_iters=100, verbose=False, c_punt=DEFAULT_C_PUNT):
        '''
        Play out a certain number of games, each time updating our win and play
        counts for any state that we visit during the game. As we continue to
        play, num_wins / num_plays for a given state should begin to converge on
        the true optimality of a state

        Args:
            g (Game): Game to train on
            num_iters (int): Number of search iterations
            verbose (bool): Whether or not to render a progress bar
            c_punt (float): The degree of exploration. Defaults to 1.4
        '''
        if verbose:
            for _ in tqdm(range(num_iters)):
                self.execute_episode(g, c_punt=c_punt)
        else:
            for _ in range(num_iters):
                self.execute_episode(g, c_punt=c_punt)

    def execute_episode(self, g, c_punt=DEFAULT_C_PUNT):
        '''
        Execute a single iteration of the search and update the internal state
        based on the generated examples

        Args:
            g (Game): The game
            c_punt (float): The degree of exploration. Defaults to 1.4
        '''
        examples = self.search_episode(g, c_punt=c_punt)
        self.update(examples)

    def search_episode(self, g, c_punt=DEFAULT_C_PUNT):
        '''
        We play a game by starting in the boards starting state and then
        choosing a random move. We then move to the next state, keeping
        track of which moves we chose. At the end of the game we go through
        our visited list and update the values of wins and plays so that we
        have a better understanding of which states are good and which are bad

        Args:
            g (Game): Game to search
            c_punt (float): The degree of exploration. Defaults to 1.4

        Returns:
            list: List of examples where each entry is of the format
                :code:`[player, state_hash, reward]`
        '''
        s = g.initial_state()
        p = 0
        examples = []
        while True:
            # Update visited with the next state
            a, expand = self.monte_carlo_action(g, s, p, c_punt)
            s = g.next_state(s, a, p)
            examples.append([p, g.to_hash(s), None])
            p = 1 - p

            if g.terminal(s):
                examples = assign_rewards(examples, g.winner(s))
                return examples

            if expand:
                # Do a random playout until we reach a terminal state
                winner = self.random_playout(g, s, p)
                examples = assign_rewards(examples, winner)
                return examples

    def monte_carlo_action(self, g, s, p, c_punt):
        '''
        Choose an action during self play based on the UCB1 algorithm. Instead of just
        choosing the action that led to the most wins in the past, we choose the action
        that balances this concern with exploration

        Args:
            g (Game): The game
            s (any): The state of the game
            p (int): The player who is about to make a move
            c_punt (float): The degree of exploration

        Returns:
            tuple: Tuple :code:`(best_move, expand)`, where playout is a boolean denoting
                whether or not the expansion phase has begun
        '''
        actions = g.action_space(s)
        expand = False

        # Stop out early if there is only one choice
        if len(actions) == 1:
            return actions[0], False

        next_state_hashes = [g.to_hash(g.next_state(s, a, p)) for a in actions]
        best_move = None

        # We first check that this player has been in each of the subsequent states
        # If they have not, then we simply choose a random action
        if all((p, s_hash) in self.plays for s_hash in next_state_hashes):

            log_total = math.log(
                sum(self.plays[(p, s_hash)] for s_hash in next_state_hashes))
            values = [
                (self.wins[(p, s_hash)] / self.plays[(p, s_hash)]) +
                c_punt * math.sqrt(log_total / self.plays[(p, s_hash)])
                for s_hash in next_state_hashes
            ]

            next_move_index = values.index(max(values))
            best_move = actions[next_move_index]
        else:
            best_move = choice(actions)
            expand = True

        return (best_move, expand)

    def update(self, examples):
        '''
        Backpropagate the result of the training episodes

        Args:
            examples (list): List of examples where each entry is of the format
                :code:`[player, state_hash, reward]`
        '''
        for [p, s, reward] in examples:
            self.plays[(p, s)] = self.plays.get((p, s), 0) + 1
            self.wins[(p, s)] = self.wins.get((p, s), 0) + reward

    def best_action(self, g, s, p):
        '''
        Get the best action for a given player in a given game state

        Args:
            g (Game): The game
            s (state): The current state of the game
            p (int): The current player

        Returns:
            int: The best action given the current knowledge of the game
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

    @staticmethod
    def random_playout(g, s, p, max_moves=1000):
        '''
        Perform a random playout and return the winner

        Args:
            g (Game): The game
            s (any): The state of the game to start the playout from
            p (player): The player whose turn it currently is
            max_moves (int): Maximum number of moves before the function exits

        Returns:
            int: The winner of the game, or -1 if there is not one
        '''
        for _ in range(max_moves):
            a = choice(g.action_space(s))
            s = g.next_state(s, a, p)
            p = 1 - p
            if g.terminal(s):
                return g.winner(s)
        return -1
