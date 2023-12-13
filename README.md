# ms-pacman-rl
Learning to play Ms. Pac-Man with reinforcement learning.

Playing Ms. Pac-Man involves handling risk and reward trade-offs in simultaneously pursuing additional points while evading ghosts. To be successful, players must develop a strategy that incorporates both the ghosts’ positions in 2D space as well as the proximity of rewards. I begin by training baseline reinforcement learning agents that follow random, proximal policy optimization (PPO), and Deep Q-Network (DQN) policies. Then I implement modifications to experiment with the impacts of gray-scaling, action trimming, reward hacking, and random starts in this arcade setting. Taken together, these modifications provide insights into strategies that succeed and fail when teaching an agent to play Ms. Pac-Man.

Please explore the following resources to learn about my experiments and results (plots and videos):
  - Read the paper: [Learning_to_Play_Ms_PacMan_with_Reinforcement_Learning.pdf](https://github.com/Zesakof/ms-pacman-rl/blob/main/Learning_to_Play_Ms_PacMan_with_Reinforcement_Learning.pdf)
  - Watch the videos: https://www.youtube.com/channel/UCn8eGj-48YBCl4eexOw30Qw

To-do in this repo:
- Hard-coded "output" folder in the .gitignore
- Using the notebook/script
  - Train and eval/plot commands with their arguments
- Dockerfile description
- Rundown of helper scripts
