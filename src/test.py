import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from model.vin import VIN
from components.gridworld import GridWorld
from components.obstacles import obstacles
from components.sample_trajectory import sample_trajectory


def visualize(dom, states_xy, pred_traj, correct):
    fig, ax = plt.subplots()
    # implot = plt.imshow(dom, cmap="Greys_r")
    ax.imshow(dom, cmap="Greys_r")
    ax.plot(states_xy[:, 0], states_xy[:, 1], c='b', label='Optimal Path')
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], '-X', c='r', label='Predicted Path')
    ax.plot(states_xy[0, 0], states_xy[0, 1], '-o', label='Start')
    ax.plot(states_xy[-1, 0], states_xy[-1, 1], '-s', label='Goal')
    legend = ax.legend(loc='upper right', shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-small')   # The legend text size
    for label in legend.get_lines():
        label.set_linewidth(0.5)        # The legend line width
    fig.savefig('./images/result{}.pdf'.format(correct))
    plt.close()


def create_map(args):
    flag_map = 1
    # set goal
    goal = [np.random.randint(args.dom_size),
            np.random.randint(args.dom_size)]
    # generate obstacle map
    obs = obstacles([args.dom_size, args.dom_size],
                    goal, args.max_obs_size)
    n_obs = obs.add_n_rand_obs(args.max_obs_num)
    # add border to map
    border_res = obs.add_border()
    # ensure whether valid map or not
    if n_obs == 0 or not border_res:
        flag_map = 0
    # get final map
    im = obs.get_final()
    return im, goal, flag_map


def predict_trajectory(im, goal, G, value_prior, states_xy, device):
    # Get number of steps to goal
    predicted_state_max_len = len(states_xy) * 2
    # Allocate space for predicted steps
    pred_traj = np.zeros((predicted_state_max_len, 2))
    # Set starting position
    pred_traj[0, :] = states_xy[0, :]
    for j in range(1, predicted_state_max_len):
        # Transform current state data
        current_state = pred_traj[j - 1, :]
        # Transform domain to Networks expected input shape
        image = 1 - im
        image_data = image.reshape(
            1, 1, args.dom_size, args.dom_size)
        value_data = value_prior.reshape(
            1, 1, args.dom_size, args.dom_size)
        # Get inputs as expected by network
        X = torch.from_numpy(
            np.concatenate((image_data, value_data), axis=1))
        S1 = torch.from_numpy(current_state[0].reshape([1, 1]))
        S2 = torch.from_numpy(current_state[1].reshape([1, 1]))
        # Get input batch
        X, S1, S2 = [d.float().to(device) for d in [X, S1, S2]]
        # Forward pass in our neural net
        _, predictions = vin(X, S1, S2, args.num_vi)
        _, predicted_action = torch.max(
            predictions, dim=1, keepdim=True)
        predicted_action = predicted_action.item()
        # Transform prediction to indices
        current_state_ind = G.map_ind_to_state(
            current_state[0], current_state[1])
        next_state = G.sample_next_state(
            current_state_ind, predicted_action)
        next_state_x, next_state_y = G.get_coords(next_state)
        pred_traj[j, 0] = next_state_x
        pred_traj[j, 1] = next_state_y
        if next_state_x == goal[0] and next_state_y == goal[1]:
            # We hit goal so fill remaining steps
            pred_traj[j + 1:, 0] = next_state_x
            pred_traj[j + 1:, 1] = next_state_y
            break
    return pred_traj


def path_planning(model, args):
    # automatically select device to make the code device agnostic
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Correct vs total:
    correct, total = 0.0, 0.0
    n_dom = 0
    for n_dom in range(args.dom_num):
        im, goal, flag_map = create_map(args)
        if flag_map == 0:
            continue
        # Generate gridworld from obstacle map
        G = GridWorld(im, goal[0], goal[1])
        # Get value prior
        value_prior = G.get_reward_prior()
        # Sample random trajectories to our goal
        states_xy, states_one_hot = sample_trajectory(G, args.traj_num)
        for i in range(args.traj_num):
            if len(states_xy[i]) > 1:
                pred_traj = predict_trajectory(
                    im, goal, G, value_prior, states_xy[i], device)
                # Plot optimal and predicted path (also start, end)
                if pred_traj[-1, 0] == goal[0] and pred_traj[-1, 1] == goal[1]:
                    correct += 1
                total += 1
                if args.plot is True:
                    visualize(G.image.T, states_xy[i], pred_traj, correct)
        sys.stdout.write("\r" + 'Progress: ' +
                         str(int((n_dom / args.dom_num) * 100)) + "%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    print('Rollout Accuracy: {:.2f}%'.format(100 * (correct / total)))


def set_args():
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='data/vin_8x8.pth', help='Path to trained weights')
    parser.add_argument('--dom_size', type=int,
                        default=8, help='size of domain')
    parser.add_argument("--dom_num", "-nd", type=int,
                        help="number of domains", default=100)
    parser.add_argument("--max_obs_num", "-no", type=int,
                        help="maximum number of obstacles", default=30)
    parser.add_argument("--max_obs_size", "-os", type=int,
                        help="maximum obstacle size", default=None)
    parser.add_argument("--traj_num", "-nt", type=int,
                        help="number of trajectories", default=1)
    parser.add_argument('--num_vi', type=int,
                        default=10, help='number of Value Iterations')
    parser.add_argument('--num_input', type=int,
                        default=2, help='number of channels in input layer')
    parser.add_argument('--num_hidden', type=int,
                        default=150, help='number of channels in first hidden layer')
    parser.add_argument('--num_qlayer', type=int,
                        default=10, help='number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='batch size')
    parser.add_argument('--plot', action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    # set args
    args = set_args()

    # instantiate VIN model
    vin = VIN(args)

    # load model parameters
    vin.load_state_dict(torch.load(args.weights))

    # path planning
    path_planning(model=vin, args=args)
