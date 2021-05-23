from Node import Node
import numpy as np
import copy


class MCTS:
    """
    Monte Carlo Tree Search algorithm
    """
    def __init__(self, board, agent, args):
        self.env_mcts = board
        self.model = agent
        self.args = args

    def run(self):

        to_play = self.env_mcts.board.turn
        original_ENV = self.env_mcts

        root = Node(0, to_play)

        value, action_probs = self.model(np.expand_dims(self.env_mcts.layer_board, axis=0))
        # Find legal moves
        action_probs = np.reshape(np.squeeze(action_probs), (64, 64))
        action_probs = action_probs * self.env_mcts.project_legal_moves()
        # Normalise to make it probability
        action_probs /= np.sum(action_probs)
        # Expnad node to childrens

        root.expand(original_ENV, to_play, action_probs)

        # Set number of runs on MCTS
        for i in range(self.args['num_simulations']):
            # for _ in range(search_depth):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = node.select_child()
                # if depth == 0:
                search_path.append(node)  # best path! NOT all options

            # Parent of the latest child
            parent = search_path[-2]

            ENV = parent.ENV
            # Now we're at a leaf node and we would like to expand

            self.env_temp = copy.deepcopy(ENV)

            move_from = action // 64
            move_to = action % 64
            move = [x for x in self.env_temp.board.generate_legal_moves() if \
                    x.from_square == move_from and x.to_square == move_to][0]

            gameover, reward = self.env_temp.step(move)

            if gameover is False:
                # If the game has not ended:
                # EXPAND
                if self.env_temp.board.turn is False:
                    self.env_temp.flip_layer_board()
                next_state = self.env_temp.layer_board
                value, action_probs = self.model(np.expand_dims(next_state, axis=0))

                action_probs = np.reshape(np.squeeze(action_probs), (64, 64))
                action_probs = action_probs * self.env_temp.project_legal_moves()
                # Normalise to make it probability
                action_probs /= np.sum(action_probs)

                node.expand(self.env_temp, parent.to_play * -1, action_probs)

            self.backpropagate(search_path, value, parent.to_play * -1)

        return root

    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
