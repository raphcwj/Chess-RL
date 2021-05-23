from chess.pgn import Game
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import copy


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return list(e_x / e_x.sum())

class DDQN_Engine(object):
    """
    DDQN Engine to train the agent's network through a MCTS for each move
    """
    def __init__(self, agent, env, opponent_random=False, memsize=1000):
        """
        Reinforce object to learn real chess
        Args:
            agent: The agent playing the chess game as white
            env: The environment including the python-chess board
            memsize: maximum amount of games to retain in-memory
        """
        self.agent = agent
        self.env = env
        self.memory = []
        self.memsize = memsize
        self.reward_trace = []
        self.memory = []
        self.sampling_probs = []
        self.fixed_agent = agent
        self.delayed_agent = agent
        self.opponent_random = opponent_random

    def learn(self, training_games=100, network_update=1,
              opponent_update=50, max_moves=25, current_level=0,
              path_to_save="multisearch"):
        """
        Run the DDQN algorithm.
        Args:
            training_games: int
                amount of games to train
            network_update: int
                update the network every c games
            opponent_update: int
                update the opponent network every c games
            max_moves: int
                maximum moves for game cutoff
            current_level: int
                current number of games that the agent has been trained for already
            path_to_save: str
                path to save model
        Returns:
            final pgn: PGN
                final game recording (use this to view final training game)
            trained agent: Agent class
                final trained agent
            final env: Board class
                final environment of last game
            training results: pandas dataframe
                record of training results

        """
        endgame_material = []
        rewards = []
        with tqdm(total=training_games, position=0, leave=True) as pbar:
            for game in range(training_games):
                # update model network every c games
                if game % network_update == 0:
                    self.fixed_agent.model = self.agent.copy_model()
                if game % opponent_update == 0:
                    self.delayed_agent.model = self.agent.copy_model()
                ### COMMENT IN IF RUNNING IN G COLAB AND YOU WANT TO SAVE MODEL CHECKPOINTS ###
                # if (game != 0) & (game % 5 == 0):
                #     projectPath = "/content/drive/My Drive/Colab Notebooks/Reinforcement Learning/Chess Project/MCTS Agent/" + path
                #     self.agent.model.save(projectPath + '/' + 'SelfPlayMCTS_checkpoint' + str(game + current_level))
                self.env.reset()
                reward, _, total_loss, turncount = self.play_game(maxiter=max_moves)
                self.env.init_layer_board()  # get latest board representation
                endgame_material.append(
                    self.env.get_material_value()[1])  # grab material value, not the "no pieces" boolean
                rewards.append(reward)
                print("End Result = ", self.env.board.result())
                print(self.env.board)
                pbar.update()
                pbar.set_postfix({"Mean Reward": np.mean(rewards),
                                  "Total Rewards.": np.sum(rewards),
                                  "Mean Adv.": np.mean(endgame_material),
                                  "Last Adv.": endgame_material[-1],
                                  "Loss": total_loss,
                                  "Turns": turncount
                                  })
                tf.keras.backend.clear_session()
                # print("Total Loss = ", total_loss)

        pgn = Game.from_board(self.env.board)
        reward_smooth = pd.DataFrame(self.reward_trace)
        reward_smooth.rolling(window=10, min_periods=0).mean().plot()  ### EDIT THIS ###

        return pgn, self.agent, self.env, reward_smooth

    def play_game(self, maxiter=25):
        """
        Play a training game of chess
        Args:
            maxiter: int
                Maximum number of steps per game
        Returns:
            reward,
            self.env.board: Board class
                board state at end of game
            total_loss: float
                loss from network update
            turncount: int
                number of turns (double of typical way of counting chess moves)
        """
        episode_end = False
        turncount = 0

        # Play one game of chess
        while not episode_end:
            state = self.env.layer_board

            # White's turn
            if self.env.board.turn:
                move_from, move_to, move, root, best_value = self.fixed_agent.get_MCTS_move(copy.deepcopy(self.env))
                # print("White move = ", move)
                action_probs = [0 for x in range(4096)]
                for k, v in root.children.items():
                    action_probs[k] = np.float(v.value())
                action_probs = softmax(action_probs)

            # Black's turn
            else:
                if self.opponent_random:
                    move = self.env.get_random_action()
                else:
                    self.env.flip_layer_board()
                    move_from, move_to, move = self.delayed_agent.get_best_move(self.env.layer_board, self.env)

            episode_end, reward = self.env.step(move)

            # Overwrite reward as white agent
            if self.env.board.result() == "1-0":
                reward = 1
            elif self.env.board.result() == "0-1":
                reward = -1
            elif self.env.board.result() == "1/2-1/2":
                reward = 0
            else:
                reward = best_value

            value = reward

            # Manage Replay Buffer
            if len(self.memory) > self.memsize:
                self.memory.pop(0)
                self.sampling_probs.pop(0)

            # Store state, moves, reward, new_state if white move
            if self.env.board.turn:
                self.memory.append([state, (move_from, move_to), value, action_probs])
                self.sampling_probs.append(1)
                self.reward_trace.append(value)
                total_loss = self.update_agent(turncount + 1)

            turncount += 1

            # End episode if turncount exceed max turns
            if turncount > maxiter:
                episode_end = True
                reward = 0

        return reward, self.env.board, total_loss, turncount

    def sample_memory(self, turncount):
        """
        Get a sample from memory for experience replay
        Args:
            turncount: int
                turncount limits the size of the minibatch
        Returns: tuple
            a mini-batch of experiences (list)
            indices of chosen experiences
        """
        minibatch = []
        memory = self.memory[:-turncount]
        probs = self.sampling_probs[:-turncount]
        sample_probs = [probs[n] / np.sum(probs) for n in range(len(probs))]
        indices = np.random.choice(range(len(memory)), min(1028, len(memory)), replace=True, p=sample_probs)
        for i in indices:
            minibatch.append(memory[i])

        return minibatch, indices

    def update_agent(self, turncount):
        """
        Update the agent using experience replay. Set the sampling probs using td error
        (memories with higher td_errors will be given a higher sampling probability)
        (this gives the agent the chance to correct erroneous predictions)
        Args:
            turncount: int
                Number of turns played. Only sample the memory if there are sufficient samples
        Returns:
        """
        if turncount < len(self.memory):
            # print("sanity check")
            minibatch, indices = self.sample_memory(turncount)
            td_errors, total_loss = self.agent.network_update(minibatch)
            for n, i in enumerate(indices):
                self.sampling_probs[i] = np.abs(td_errors[n])
        else:
            total_loss = np.inf
        return total_loss


if __name__ == '__main__':
    from Board import Board
    from Agent import Agent
    board = Board()
    agentMCTS = Agent(initial_network="MCTS", gamma=0.1, lr=0.1, args={"num_simulations":100})
    DDQN = DDQN_Engine(agent=agentMCTS, env=board, opponent_random=True)
    print("Engine class initialized")
    print(DDQN)

