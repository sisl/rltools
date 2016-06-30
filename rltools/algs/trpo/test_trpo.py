import gym 
import prettytensor as pt
from rltools.algs import TRPOSolver
from rltools.utils import simulate

env = gym.make("CartPole-v0")

# define sime parameters, look at trpo source to see the defaults
config = {}
config["train_iterations"] = 100 # number of trpo iterations
config["max_pathlength"] = 300 # maximum length of an env trajecotry
config["gamma"] = 0.99 # discount facotr

solver = TRPOSolver(env, config = config)
solver.learn()

simulate(env, solver, 100, render=True)
