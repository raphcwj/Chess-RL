import chess
import numpy as np

mapper = {"p": 0, "r": 1, "n": 2, "b": 3, "q": 4, "k": 5, "P": 0, "R": 1, "N": 2, "B": 3, "Q": 4, "K": 5}


class Board(object):

    def __init__(self, FEN=None):
        """
        Chess Board Environment
        Args:
            FEN: str
                Initialized board in FEN notation, if None then start in the default chess position
        """
        self.FEN = FEN
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.init_action_space()
        self.layer_board = np.zeros(shape=(8, 8, 8))
        self.init_layer_board()

    def init_action_space(self):
        """
        Initialize the action space
        Returns:
        """
        self.action_space = np.zeros(shape=(64, 64))

    def init_layer_board(self):
        """
        Initalize the numerical representation of the environment
        This representation will be fed into the neural network
        Returns:
        """
        self.layer_board = np.zeros(shape=(8, 8, 8))
        for i in range(64):
            row = i // 8
            col = i % 8
            piece = self.board.piece_at(i)
            if piece == None:
                continue
            elif piece.symbol().isupper():
                sign = 1
            else:
                sign = -1

            layer = mapper[piece.symbol()]
            # layers 0 - 5 represent pieces
            self.layer_board[layer, row, col] = sign

        # if it is white's turn, layer 6 = 1/full_movenumber, else 0
        if self.board.turn:
            # keep track of the inverse of move number
            self.layer_board[6, :, :] = 1 / self.board.fullmove_number

            # if a draw can be claimed, layer 7 = 1, else 0
        if self.board.can_claim_draw():
            self.layer_board[7, :, :] = 1

    def step(self, action):
        """
        Run a step
        Args:
            action: tuple of 2 integers
                Move from, Move to
        Returns:
            epsiode end: Boolean
                Whether the episode has ended
            reward: int
                1 if checkmate, else 0
        """

        self.board.push(action)  # make move
        self.init_layer_board()  # new board representation
        # Checkmate reward as white agent
        reward = self.board.is_checkmate()

        # if game is over due to checkmate, stalemate, etc
        if self.board.is_game_over():
            episode_end = True
        else:
            episode_end = False

        return episode_end, reward

    def get_random_action(self):
        """
        Sample a random action
        Returns: move
            A legal python chess move.
        """
        legal_moves = [x for x in self.board.generate_legal_moves()]
        legal_moves = np.random.choice(legal_moves)
        return legal_moves

    def project_legal_moves(self):
        """
        Create a mask of legal actions
        Returns: np.ndarray with shape (64,64)
        """
        self.action_space = np.zeros(shape=(64, 64))
        moves = [[x.from_square, x.to_square] for x in self.board.generate_legal_moves()]
        # create a 64x64 mathematical representation for actions
        for move in moves:
            # all legal move-from and move-to tiles will equal 1
            self.action_space[move[0], move[1]] = 1
        return self.action_space

    def get_material_value(self):
        """
        Sums up the material balance using Reinfield values
        Returns: The material balance on the board
        """
        white_sign = 1
        black_sign = -1

        white_value = self._get_value(white_sign)
        black_value = self._get_value(black_sign)

        material_value = white_value - black_value

        if (white_value == 0) | (black_value == 0):
            no_pieces = True
        else:
            no_pieces = False

        return no_pieces, material_value

    def _get_value(self, player_sign):
        """
        Function to calculate material value of a given player.
        """
        pawns = 1 * np.sum(self.layer_board[0, :, :] == player_sign)
        rooks = 5 * np.sum(self.layer_board[1, :, :] == player_sign)
        minor = 3 * np.sum(self.layer_board[2:4, :, :] == player_sign)
        queen = 9 * np.sum(self.layer_board[4, :, :] == player_sign)
        material_value = pawns + rooks + minor + queen
        return material_value

    def reset(self):
        """
        Reset the environment
        Returns:
        """
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.init_layer_board()
        self.init_action_space()

    def flip_layer_board(self):
        """
        Flips mathematical representation of board so that black agents see the board in the same perspective

        """
        self.layer_board[0:5, :, :] = self.layer_board[0:5, ::-1, :] * -1


if __name__ == '__main__':
    print("Board class initialized")
    board = Board()
    print(board.board)
