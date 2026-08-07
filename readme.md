# Training an AI agent to play Centipede with Reinforcement Learning and a Convolution Neural Network
Andrew Metzroth
Final project, CSPB 3202 at CU Boulder


### Overview
This project is an introductory dive into using various types of Q-learning and Reinforcement learning techniques to teach AI agents to play the Atari version of Centipede in Gymnasium's Arcade Learning environment. 

Huge thanks to the tensorflow tutorial on actor-critic neural networks for getting me started:  https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic. Additionally, huge thanks to my classmate Thomas Dunn for both moral support throughout the project and a bunch of troubleshooting assistance as I tried to get the appropriate packages loaded onto my computer!

In the repository you should find:

- centipede_cnn_pooling.py is the primary code base for the project. This is where I worked and refined the most

- centipede_cnn.py reflects an early version of the neural network without pooling.

- centipede_cnn_pooling_penalties.py represents an attempt to add penalties for repeated movements. It did not change behavior substantially.

- centipede_qagent.py and centipede_random_agent.py explore limited agents that do not use reinforcement learning techiniques.

- Final_report.ipynb and Final_report.html provide a narrative about my approach to the project and attempt to solve the problem.

- In the folder centipede_agents, Various videos of RL aznd Q agent performance, labeled by type of agent and length of training.
