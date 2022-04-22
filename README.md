# Udacity-Continuous-control 
__Project Overview:__
This project applies Deep Reinforcement Learning on the Udacity continuous control project

![reacher](results/reacher.gif)

In this environment, a double-jointed arm can move to target locations. The environment is considered solved if a reward of +100 is obtain for 30 consecutive episodes.

The method to use is an actor-critic algorithm, the Deep Deterministic Policy Gradients (DDPG) algorithm.

### Reward
A reward of +0.1 is provided for each step that the agent's hand is in the goal location.

### State space

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.

### Action space

Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


The Environment Follow the instructions below to explore the environment on your own machine! You will also learn how to use the Python API to control your agent.

__Step 1: Clone the DRLND Repository__ 

Follow the instructions in the [DRLND GitHub repository to set up your Python environment](https://github.com/udacity/deep-reinforcement-learning#dependencies) . These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.


__Step 2: Download the Unity Environment__

For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip )

Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip )

Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip )

Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip) 

Then, place the file in the p2_continuous-control/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.


__Step 3: Explore the Environment__
After you have followed the instructions above, open Continuous_control.ipynb (located in the p1_continuous-control/ folder in the DRLND GitHub repository) and experiment to learn how to use the Python API to control the agent.






