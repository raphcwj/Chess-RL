# Chess-RL
Raphael Chew, Shaun Fendi Gan

(MIT 6.884: Computational Sensorimotor Learning)

DeepMind created their groundbreaking AlphaZero algorithm with an estimated $35 million dollars of computational powwer. The goal of this project was to explore whether it would be possible to achieve some headway with a $9.99 Google Colab Pro GPU. 


## Offline Reinforcement Learning: Training a Lean Chess Agent

Our research experiments with lighter versions of AlphaZero's offline reinforcement learning algorithm for chess. We reconstruct a leaner MCTS value and policy network algorithm from scratch, to investigate the possibility of training a less capable chess agent, but within the computational limitations of the average machine learning engineer. Specifically, we investigate the viability of using lean CNN architectures for mimicking the values and policies discovered by the MCTS during self-play. 

## Abstract
Lean Convolutional Neural Networks (CNNs) were trained in a Double Deep Q-Network (DDQN) setup using Reinforcement Learning (RL) to mimic a Monte Carlo Tree Search (MCTS) algorithm at playing chess. In just 120 training games, this agent achieved a 10.2% ± 3.8% win rate and <1\% loss rate against an opponent making random moves. Similarly, semi-supervised methods achieved a $15.3% ± 6.4% win rate with <1% loss rate against the same opponent from 120 training games. Reward shaping and behavior cloning were also tested but did not produce effective chess agents. 
