# Chess-RL
Raphael Chew, Shaun Fendi Gan

(MIT 6.884: Computational Sensorimotor Learning)

DeepMind created their groundbreaking AlphaZero algorithm with an estimated $25 million dollars of computational power [1]. The goal of this project was to explore whether it would be possible to achieve some headway with a $9.99 Google Colab Pro GPU. 

## Offline Reinforcement Learning: Training a Lean Chess Agent

Our research experiments with lighter versions of AlphaZero's offline reinforcement learning algorithm for chess. We reconstruct a leaner MCTS value and policy network algorithm from scratch, to investigate the possibility of training a less capable chess agent, but within the computational limitations of the average machine learning engineer. Specifically, we investigate the viability of using lean CNN architectures for mimicking the values and policies discovered by the MCTS during self-play. 

## Abstract
Lean Convolutional Neural Networks (CNNs) were trained in a Double Deep Q-Network (DDQN) setup using Reinforcement Learning (RL) to mimic a Monte Carlo Tree Search (MCTS) algorithm at playing chess. In just 120 training games, this agent achieved a 10.2% ± 3.8% win rate and < 1% loss rate against an opponent making random moves. Similarly, semi-supervised methods achieved a $15.3% ± 6.4% win rate with < 1% loss rate against the same opponent from 120 training games. Reward shaping and behavior cloning were also tested but did not produce effective chess agents. 

## Files 
- `Board.py`- Board environment class for mathematically representing the chess board, stepping through moves, and checking game results
- `Agent.py` - Agent actor class for initializing networks, updating networks, and selecting moves
- `DDQN_Engine.py` - DDQN engine class for training agent, updating agent, and managing the memory buffer
- `MCTS.py` - MCTS algorithm class for searching promising move paths and backpropagating board state values
- `MCTS_Stockfish` - Stockfish-guided MCTS algorithm class for added demonstration learning by providing expert board state values
- `Node.py` - Node class to hold relevant attributes (value scores, visit counts, child nodes) for MCTS; contains UCB score function
- `GameSweep.py` - Game Sweep class to evaluate agent under test conditions, save pgn, and show game gif
- `Trained Agent` - Contains a 120-game trained instance of the DDQN-MCTS agent (1-layer CNN, poisson loss, lr = 0.1)

## Training a DDQN-MCTS Agent
```python
from Board import Board
from Agent import Agent
from DDQN_Engine import DDQN_Engine


# Parameters for training
agent = "MCTS"  # initialize new 1-layer CNN MCTS agent
current_level = 0  # number of games trained for current model
num_sims = 100  # number of MCTS simulations per move search
path = None  # path to save model to

args = {'num_simulations': num_sims}

board = Board()
agent = Agent(initial_network=agent, gamma=0.1, lr=0.1, args=args)
engine = DDQN_Engine(agent=agent, env=board, opponent_random=True)
pgn, trained_agent, final_env, reward_df = engine.learn(training_games=300,
                                                        network_update=1,
                                                        opponent_update=50,
                                                        max_moves=150,
                                                        current_level=0,
                                                        path_to_save=path)

```

## Report
|<a href="https://github.com/raphcwj/Chess-RL/blob/main/Paper/Chess-RL%20Paper.pdf"><img src="https://github.com/raphcwj/Chess-RL/blob/main/Paper/Chess-RL%20Paper%20Thumbnail.png" alt="Illustration" width="220px"/></a>|
|:--:|
|Full Report|

## Acknowledgements
Shout out to Arjan Groen for building Q-Learning engines for Capture Chess and Real Chess

[1] Gibney, Elizabeth. "Self-taught AI is best yet at strategy game Go". Nature News. October 2017. Retrieved 10 May 2020.
