This project implements Q-Learning algorithms to navigate a robot through a grid-based environment. It includes the basic Tabular Q-Learner, an enhanced Double Q-Learner, and the Dyna-Q algorithm for simulated experiences.

Key Components

	1.	TabularQLearner: Implements the core Q-Learning algorithm, updating Q-values based on actions and rewards. It supports Dyna-Q with simulated experiences.
	2.	DoubleQLearner: Enhances Q-Learning by using two Q-tables to reduce overestimation bias.
	3.	Main Script: Runs experiments, visualizes results, and evaluates the learner’s performance.

How Q-Learning Works

	•	Q-Learning is a model-free reinforcement learning algorithm used to find the optimal action-selection policy. It updates the Q-values (which represent the value of taking an action in a state) based on the received reward and the estimated future rewards.
	•	TabularQLearner:
	•	Training: Updates Q-values using the formula:

Q(s, a) = (1 - alpha) x Q(s, a) + alpha x (r + gamma x max_a Q(s', a))

where alpha is the learning rate, gamma is the discount factor, and r is the reward.
	•	Dyna-Q: Enhances learning by generating simulated experiences based on past experiences, which helps in faster convergence.
	•	DoubleQLearner: Uses two separate Q-tables to mitigate the overestimation bias present in traditional Q-Learning, improving stability.
