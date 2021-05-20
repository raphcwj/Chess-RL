from stockfish import Stockfish	
from Node import Node
import numpy as np
import copy

class MCTS:

    def __init__(self, board, agent, args):
        self.env_mcts = board
        self.model = agent
        self.args = args

    def run(self):
        # print("White inner env = ", self.env_mcts)
        # print("INNER white legal moves",[x for x in self.env_mcts.board.generate_legal_moves()])
        to_play = self.env_mcts.board.turn
        original_ENV = self.env_mcts

        root = Node(0, to_play)
        
        value, action_probs = self.model(np.expand_dims(self.env_mcts.layer_board,axis=0))

        M = 10000
        gamma = 50 # decay rate
        stockfish.set_fen_position(self.env_mcts.board.fen())
        value_dict = stockfish.get_evaluation()  # {'type': 'cp' or 'mate, 'value' : 9405 or 1 (-1 for black) ( mate in 1 )}      
        value = -M*np.exp((value_dict['value'])/gamma) if value_dict['type'] == 'mate' else value_dict['value']
        value /= 10000


        # Find legal moves
        action_probs = np.reshape(np.squeeze(action_probs), (64, 64))
        action_probs = action_probs* self.env_mcts.project_legal_moves()
        # Normalise to make it probability
        action_probs /= np.sum(action_probs)
        # Expnad node to childrens

        root.expand(original_ENV, to_play, action_probs)
        
        # Set number of runs on MCTS
        for i in range(self.args['num_simulations']):
        # for _ in range(search_depth):
            node = root
            search_path = [node]

            depth = 0
            while node.expanded():
                action, node = node.select_child()
                # if depth == 0: 
                search_path.append(node) # best path!! NOT all options
            
            # Parent of the latest child
            parent = search_path[-2] 
            
            ENV = parent.ENV
            # Now we're at a leaf node and we would like to expand

            self.env_temp = copy.deepcopy(ENV)

            # action_index = np.array([, action % 64 ])
            move_from =  action // 64 
            move_to =  action % 64
            move = [x for x in self.env_temp.board.generate_legal_moves() if \
                    x.from_square == move_from and x.to_square == move_to][0]
            
            # TODO black should be on left, add additional agent?
            gameover, reward = self.env_temp.step(move)

            if gameover is False:
                # If the game has not ended:
                # EXPAND
                if self.env_temp.board.turn == False:
                  self.env_temp.flip_layer_board()
                next_state = self.env_temp.layer_board
                _, action_probs = self.model(np.expand_dims(next_state,axis=0))
        
                action_probs = np.reshape(np.squeeze(action_probs), (64, 64))
                action_probs = action_probs* self.env_temp.project_legal_moves()
                # Normalise to make it probability
                action_probs /= np.sum(action_probs)
    
                node.expand(self.env_temp, parent.to_play * -1, action_probs)
            
            stockfish.set_fen_position(self.env_temp.board.fen())
            value_dict = stockfish.get_evaluation()
            # Black's turn next
            if self.env_temp.board.turn == False:
                # Black's turn, perspective as white?
                value = M*np.exp((-value_dict['value'])/gamma) if value_dict['type'] == 'mate' else value_dict['value']
                value /= 10000

            else:
                # Get board as perspective of black, 
                value = -M*np.exp((value_dict['value'])/gamma) if value_dict['type'] == 'mate' else value_dict['value']
                value /= 10000

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

def init_stockfish(fen ="rnbqkbnr/1ppppppp/8/p6Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 0 3"):
    stockfish = Stockfish("/content/drive/My Drive/Colab Notebooks/Reinforcement Learning/Chess Project/stockfish/stockfish_13_linux_x64")
    stockfish.set_fen_position(fen)
    stockfish.set_elo_rating(2600)
    return stockfish


if __name__ == "__main"":
	# !pip install stockfish
	stockfish = init_stockfish()
