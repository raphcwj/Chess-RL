import numpy as np
import pandas as pd
from tqdm import tqdm
import pgn2gif
from IPython.display import Image
from chess.pgn import Game


class GameSweep_SelfPlay(object):

    def __init__(self, WhiteAgent, BlackAgent, env, max_moves=25):
        """
        Reinforce object to learn capture chess
        Args:
            agent: The agent playing the chess game as white
            env: The environment including the python-chess board
            memsize: maximum amount of games to retain in-memory
        """
        self.WhiteAgent = WhiteAgent
        self.BlackAgent = BlackAgent
        self.env = env
        self.reward_trace = []
        self.max_moves = max_moves

    def play(self, checkMaterialValues=[25]):
        """
        Play a game of capture chess
        Args:
            checkMaterialValues: list of int
                move_counts to check material values on

        Returns:
            pgn: gameplay
            reward_smooth: smoothened rewards
            material_values: end-game material advantage
        """
        episode_end = False
        turncount = 0
        material_values = []  # list of material values at move x, y, z
        WhiteWins = 0
        BlackWins = 0
        Draws = 0
        # Play one game of chess
        while not episode_end:
            state = self.env.layer_board

            white_move_from, white_move_to, white_move = self.get_bot_move(self.WhiteAgent,
                                                                           np.expand_dims(state, axis=0))
            episode_end, pre_reward = self.env.step(white_move)
            new_state = self.env.layer_board

            if self.env.board.result() == "*":
                new_state[0:5, :, :] = -new_state[0:5, :, :]
                new_state = new_state[:, ::-1, :]
                black_from, black_to, black_move = self.get_bot_move(self.BlackAgent,
                                                                     np.expand_dims(new_state, axis=0))
                episode_end, post_reward = self.env.step(black_move)

            turncount += 1

            # Store material advantage for the required timesteps
            if turncount in checkMaterialValues:
                material_values.append(self.env.get_material_value()[1])

            # End game if turn count exceeds max moves allowed
            if turncount > self.max_moves:
                episode_end = True

        # ensure correct number of material values (append latest adv if move count is not reached)
        if (len(material_values) < len(checkMaterialValues)) & (len(material_values) > 0):
            num_missing = len(checkMaterialValues) - len(material_values)
            for i in range(num_missing):
                material_values.append(material_values[-1])
        elif len(material_values) == 0:
            material_values = [0] * len(checkMaterialValues)

        # Collect final result
        if self.env.board.result() == "1-0":
            WhiteWins = 1
        elif self.env.board.result() == "0-1":
            BlackWins = 1
        elif self.env.board.result() == "1/2-1/2":
            Draws = 1

        result = [WhiteWins, BlackWins, Draws]

        # Once game is over
        self.pgn = Game.from_board(self.env.board)
        reward_smooth = pd.DataFrame(self.reward_trace)

        return self.pgn, reward_smooth, material_values, result

    def get_bot_move(self, bot, state):
        """
        Gets the agent/opponent move
        Args:
            bot: str or Keras network
                the agent/opponent that moves
            state: 8x8x8 matrix array
                numerical state of the board
        Returns:
            move
        """
        if bot == "random":
            move = self.env.get_random_action()
            move_from = move.from_square
            move_to = move.to_square
        else:
            _, action_values = bot.predict(state)
            action_values = np.reshape(np.squeeze(action_values), (64, 64))
            # Environment determines which moves are legal
            action_space = self.env.project_legal_moves()
            # Get legal action_values
            action_values = np.multiply(action_values, action_space)

            # Get best move
            maxes = np.argwhere(action_values == action_values.max())
            move_from, move_to = maxes[np.random.randint(len(maxes))]
            moves = [x for x in self.env.board.generate_legal_moves() if
                     x.from_square == move_from and x.to_square == move_to]

            if len(moves) == 0:  # If all legal moves have negative action value, explore.
                move = self.env.get_random_action()
                move_from = move.from_square
                move_to = move.to_square
            else:
                move = np.random.choice(moves)  # If there are multiple max-moves, pick a random one.
                move_from = move.from_square
                move_to = move.to_square

        return move_from, move_to, move

    def evaluate(self, n_trials, max_moves, checkMaterialValues):
        if max_moves:
            self.max_moves = max_moves

        col_names = [f"Move #{num}" for num in checkMaterialValues]
        MaterialAdv_Overview = pd.DataFrame(columns=col_names, index=list(np.arange(n_trials)))
        WhiteWins = 0
        BlackWins = 0
        Draws = 0

        with tqdm(total=n_trials, position=0, leave=True) as pbar:
            for i in range(n_trials):
                pgn, reward_smooth, values, result = self.play(checkMaterialValues)
                MaterialAdv_Overview.loc[i] = values
                WhiteWins += result[0]
                BlackWins += result[1]
                Draws += result[2]
                self.env.reset()
                pbar.update()

        MaterialAdvStats = pd.DataFrame(columns=["MoveNumber", "Min", "Max", "Mean", "Median", "Std"])

        MaterialAdvStats["MoveNumber"] = [num for num in checkMaterialValues]
        MaterialAdvStats["Min"] = [MaterialAdv_Overview[f"Move #{num}"].min() for num in checkMaterialValues]
        MaterialAdvStats["Max"] = [MaterialAdv_Overview[f"Move #{num}"].max() for num in checkMaterialValues]
        MaterialAdvStats["Mean"] = [np.mean(MaterialAdv_Overview[f"Move #{num}"]) for num in checkMaterialValues]
        MaterialAdvStats["Median"] = [np.median(MaterialAdv_Overview[f"Move #{num}"]) for num in checkMaterialValues]
        MaterialAdvStats["Std"] = [np.std(MaterialAdv_Overview[f"Move #{num}"]) for num in checkMaterialValues]
        # Results dictionary
        Results = {"WhiteWins": WhiteWins,
                   "BlackWins": BlackWins,
                   "Draws": Draws}

        return MaterialAdv_Overview, MaterialAdvStats, Results

    def save_sweep(self, game_name="RandomGame", play_gif=True, path=None):

        # Save sweep game
        if path:
            self.sweep_pgn_game_path = path

        self.sweep_pgn_game_path = "/content/drive/My Drive/Colab Notebooks/Reinforcement Learning/Chess Project/games/" + game_name
        print(self.pgn, file=open(self.sweep_pgn_game_path + ".pgn", "w"), end="\n\n")
        creator = pgn2gif.PgnToGifCreator(reverse=False, duration=0.5, ws_color='white', bs_color='gray')
        creator.create_gif(self.sweep_pgn_game_path + ".pgn", out_path=self.sweep_pgn_game_path + ".gif")  # creates gif

        if play_gif == True:
            return Image(open(self.sweep_pgn_game_path + ".gif", 'rb').read())

    def show_gif(self):
        return Image(open(self.sweep_pgn_game_path + ".gif", 'rb').read())