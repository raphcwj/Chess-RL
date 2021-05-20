from Board import Board
from Agent import Agent
from DDQN_Engine import DDQN_Engine


# Parameters for training
agent = "MCTS"  # initialize new 1-layer CNN MCTS agent
current_level = 0  # number of games trained for current model
num_sims = 100  # number of MCTS simulations per move search
path = None  # path to save model to
args = {'num_simulations': num_sims}

# Train a DDQN-MCTS Agent for 300 games
board = Board()
agent = Agent(initial_network=agent, gamma=0.1, lr=0.1, args=args)
engine = DDQN_Engine(agent=agent, env=board, opponent_random=True)
pgn, trained_agent, final_env, reward_df = engine.learn(training_games=300,
                                                        network_update=1,
                                                        opponent_update=50,
                                                        max_moves=150,
                                                        current_level=0,
                                                        path_to_save=path)
