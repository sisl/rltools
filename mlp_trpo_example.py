import gym 
import prettytensor as pt
import tensorflow as tf

from rltools.algs import TRPOSolver
from rltools.utils import simulate
from rltools.models import softmax_mlp

env = gym.make("CartPole-v0")

# define some parameters, see trpo source to see the defaults
config = {}
config["train_iterations"] = 1 # number of trpo iterations
config["max_pathlength"] = 250 # maximum length of an env trajecotry
config["timesteps_per_batch"] = 1000
config["eval_trajectories"] = 50
config["eval_every"] = 50
config["gamma"] = 0.95 # discount factor
config["save_path"] = "results" # will create a results directory for our data

# lets initialize a model
input_obs = tf.placeholder(tf.float32, shape=(None,) + env.observation_space.shape, name="obs")
net = softmax_mlp(input_obs, env.action_space.n, layers=[32,32], activation=tf.nn.tanh)

solver = TRPOSolver(env, config=config, policy_net=net, input_layer=input_obs)
solver.learn()

simulate(env, solver, 100, render=True)
