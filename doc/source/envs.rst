Environment-Related Interfaces
==============================

Different from supervised learning where training data are readily available, reinforcement learning generates data in an online manner.
In concrete, an agent interact with one or more environments where, at each timestep, the agent observes an observation from the environment, decides which action should be taken based on the observation, and receive a reward signal from the environment.
The environment defines the task (i.e., the MDP) to be solved.
In most real-world applications, engineers/researchers are required to implement an environment, simulating the actually considered problem.
As most reinforcement learning implementations have respect OpenAI gym interface, EasyRL follows it for the convenience of our users.
