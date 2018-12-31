import math
from random import choice
from tqdm import tqdm

from gameai.core import Algorithm

DEFAULT_C_PUNT = 1.4
EPSILON = 1e-8


class MCTS(Algorithm):
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

    def get_untried_actions(self, g, s):
        '''
        Get all the moves that have not been tried yet

        Args:
            g (Game): The game
            s (any): The state of the game

        Returns:
            list: A list of actions
        '''
        actions = g.action_space(s)
        s_hash = g.to_hash(s)
        return [a for a in actions if (s_hash, a) not in self.plays]

    def search(self, g, s, p, num_iters=100, verbose=False, nnet=None, c_punt=DEFAULT_C_PUNT):
        '''
        Play out a certain number of games, each time updating our win and play
        counts for any state that we visit during the game. As we continue to
        play, num_wins / num_plays for a given state should begin to converge on
        the true optimality of a state

        Args:
            g (Game): Game to train on
            s (Game): The state to start the search in
            p (int): The current player
            num_iters (int): Number of search iterations
            verbose (bool): Whether or not to render a progress bar
            nnet (Network): Optional nework to be used for the policy instead
                of a naive random playout
            c_punt (float): The degree of exploration. Defaults to 1.4

        Returns:
            list: List of examples where each entry is of the format
                :code:`[state_hash, action, reward]`
        '''
        iter_wrapper = tqdm if verbose else lambda x: x
        examples = []
        for _ in iter_wrapper(range(num_iters)):
            iter_examples = self.search_episode(
                g, s.copy(), p, nnet=nnet, c_punt=c_punt)
            self.update(iter_examples)
            examples += iter_examples
        return examples

    def search_episode(self, g, s, p, nnet=None, c_punt=DEFAULT_C_PUNT):
        '''
        We play a game by starting in the boards starting state and then
        choosing a random move. We then move to the next state, keeping
        track of which moves we chose. At the end of the game we go through
        our visited list and update the values of wins and plays so that we
        have a better understanding of which states are good and which are bad

        Args:
            g (Game): Game to search
            s (any): The state to start on
            p (int): The current player
            nnet (Network): Optional nework to be used for the policy instead
                of a naive random playout
            c_punt (float): The degree of exploration. Defaults to 1.4

        Returns:
            list: List of examples where each entry is of the format
                :code:`[state_hash, action, reward]`
        '''
        examples = []
        while True:
            a, expand = self.monte_carlo_action(g, s, nnet, c_punt)
            s_hash = g.to_hash(s)
            examples.append([s_hash, a, None])
            s = g.next_state(s, a, p)
            p = 1 - p

            if g.terminal(s):
                examples = self.assign_rewards(examples, g.reward(s, 0))
                return examples

            # Do a random playout until we reach a terminal state. If a network
            # is provided we instead use it's predicted outcome
            if expand:
                reward = None
                if nnet:
                    _, reward = nnet.predict_single(s)
                else:
                    reward = self.random_playout(g, s, p)
                examples = self.assign_rewards(examples, reward)
                return examples

    def monte_carlo_action(self, g, s, nnet, c_punt):
        '''
        Choose an action during self play based on the UCB1 algorithm. Instead of just
        choosing the action that led to the most wins in the past, we choose the action
        that balances this concern with exploration

        Args:
            g (Game): The game
            s (any): The state of the game
            p (int): The player who is about to make a move
            nnet (Network): A network that serves as the default policy
            c_punt (float): The degree of exploration

        Returns:
            tuple: Tuple :code:`(best_move, expand)`, where playout is a boolean denoting
                whether or not the expansion phase has begun
        '''
        actions = g.action_space(s)
        untried_actions = self.get_untried_actions(g, s)
        s_hash = g.to_hash(s)

        if len(actions) == 1:
            return (actions[0], False)

        if untried_actions != []:
            if nnet:
                policy, _ = nnet.predict_single(s)
                return (policy.argmax(), True)
            else:
                return (choice(untried_actions), True)

        best_move = None
        best_ubc = -float('inf')
        log_total = math.log(sum(self.plays[(s_hash, a)] for a in actions))

        for a in actions:
            wins = self.wins[(s_hash, a)]
            plays = self.plays[(s_hash, a)]
            q_value = wins / float(plays)
            ubc = q_value + c_punt * math.sqrt(log_total / plays)
            if ubc > best_ubc:
                best_ubc = ubc
                best_move = a

        return (best_move, False)

    def best_action(self, g, s, _):
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
        untried_actions = self.get_untried_actions(g, s)
        s_hash = g.to_hash(s)

        # Stop out early if there is only one choice
        if len(actions) == 1:
            return actions[0]

        if untried_actions != []:
            return choice(actions)

        q_values = [self.wins[(s_hash, a)] / self.plays[(s_hash, a)]
                    for a in actions]
        best_move_index = q_values.index(max(q_values))
        return actions[best_move_index]

    def policy(self, g, s, temp=1):
        '''
        Return the favorability of each action in the games action space.

        Args:
            g (Game): The game
            s (any): The state to evaluate
            temp (float): Temperature of the probabilities

        Returns:
            :obj:`list` of :obj:`float`: The favorabiltiy of each action

        Examples:
            >>> g.total_action_space_size
            9
            >>> g.action_space()
            [1,3,6]
            >>> mcts.pi(g, s, temp=1)
            [0, 0.2, 0, 0.5, 0, 0, 0.3, 0, 0]
        '''
        s_hash = g.to_hash(s)
        counts = [self.wins[(s_hash, a)] / self.plays[(s_hash, a)] if (
            s_hash, a) in self.wins else 0 for a in range(g.total_action_space_size)]

        if sum(counts) == 0:
            raise ValueError('Cannot call policy before searching')

        # One hot encode
        if temp == 0:
            best_move = counts.index(max(counts))
            return [1 if i == best_move else 0 for i in range(g.total_action_space_size)]

        counts = [count**(1/temp) for count in counts]
        return [count/float(sum(counts)) for count in counts]

    def update(self, examples):
        '''
        Backpropagate the result of the training episodes. This assigns Q-values to
        each state-action pair

        Args:
            examples (list): List of examples where each entry is of the format
                :code:`[player, state_hash, reward]`
        '''
        for [s, a, reward] in examples:
            self.plays[(s, a)] = self.plays.get((s, a), 0) + 1
            self.wins[(s, a)] = self.wins.get(
                (s, a), 0) + (1 if reward > 0 else 0)

    @staticmethod
    def random_playout(g, s, p):
        '''
        Perform a random playout and return the winner

        Args:
            g (Game): The game
            s (any): The state of the game to start the playout from
            p (player): The player whose turn it currently is
            max_moves (int): Maximum number of moves before the function exits

        Returns:
            int: The reward of the game from the perspective of player 0
        '''
        while True:
            a = choice(g.action_space(s))
            s = g.next_state(s, a, p)
            p = 1 - p
            if g.terminal(s):
                return g.reward(s, 0)

    @staticmethod
    def assign_rewards(examples, reward):
        '''
        Assign rewards to the examples after the outcome is known. Note that this
        is always from the perspective of the starting player (0)

        Args:
            examples (list): List of examples where each entry is of the format
                :code:`[state_hash, action, None]`
            reward (int): The reward from the perspective of player 0

        Returns:
            list: List in the format :code:`[state_hash, action, reward]`
        '''
        return [[s_hash, a, reward] for [s_hash, a, _] in examples]
