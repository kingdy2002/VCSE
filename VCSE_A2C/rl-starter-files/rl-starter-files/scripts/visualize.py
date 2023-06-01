import argparse
import time
import numpy
import torch

import utils

import csv


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=2,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    device=device, argmax=args.argmax, use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

# Run the agent

if args.gif:
   from array2gif import write_gif
   frames = []

# Create a window to view the environment
# env.render('human')
agent_pos_visits = dict()

for episode in range(args.episodes):
    obs = env.reset()

    while True:
        # env.render('human')
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))
        
        if tuple(env.agent_pos) not in agent_pos_visits:
            agent_pos_visits[tuple(env.agent_pos)] = 0
        agent_pos_visits[tuple(env.agent_pos)] += 1
        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)

        if done: # or env.window.closed:
            break
        
    image = numpy.array(frames)
    # if env.window.closed:
    #     break

    # with open('visitations.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=' ',
    #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for elem in agent_pos_visits:
    #         writer.writerow([elem, agent_pos_visits[elem]])
    with open('environment.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        pic = frames[0].transpose(1, 2, 0)
        for i in range(pic.shape[0]):
            for j in range(pic.shape[1]):
                writer.writerow(pic[i][j])

from PIL import Image
pic = frames[0].transpose(1, 2, 0)
image = numpy.array(pic)
img = Image.new(mode='RGB', size=(pic.shape[1], pic.shape[0]), color=0x0000FF)
pix=img.load()
for i in range(pic.shape[1]):
    for j in range(pic.shape[0]):
        pix[i,j]=(pic[j][i][0],pic[j][i][1],pic[j][i][2])
img.save('./image/'+args.env+'.png', format='PNG')


