import argparse
import sys
import numpy as np
import torch
from model.vin import VIN
from component.blueprint.obstacles import obstacles
from component.blueprint.gridworld import GridWorld
from component.controller.search_path import search_path
from component.view.view import visualize_path
from component.view.view import visualize_reward
from component.view.view import visualize_v_value
np.random.seed(0)


def create_map(args):
    flag_map = 1
    goal = [np.random.randint(args.dom_size),
            np.random.randint(args.dom_size)]

    # generate obstacle map
    obs = obstacles(domsize=[args.dom_size, args.dom_size],
                    goal=goal, size_max=args.max_obs_size)
    n_obs = obs.add_n_rand_obs(args.max_obs_num)

    # add border to map
    border_res = obs.add_border()

    # ensure whether valid map or not
    if n_obs == 0 or not border_res:
        flag_map = 0

    return obs.dom, goal, flag_map


def predict_trajectory(net, GW, reward_prior, paths, device):
    predicted_state_max_len = len(paths) * 2
    pred_traj = np.zeros((predicted_state_max_len, 2))
    pred_traj[0, :] = paths[0, :]

    for j in range(predicted_state_max_len - 1):
        # Transform current state data
        domain_data = GW.domain.reshape(
            1, 1, args.dom_size, args.dom_size)
        reward_data = reward_prior.reshape(
            1, 1, args.dom_size, args.dom_size)
        domain_reward_data = torch.from_numpy(
            np.concatenate((domain_data, reward_data), axis=1))

        # current state
        current_state = pred_traj[j, :]
        state_row = torch.from_numpy(current_state[0].reshape([1, 1]))
        state_col = torch.from_numpy(current_state[1].reshape([1, 1]))

        # Get input batch
        domain_reward_data, state_row, state_col = [d.float().to(device)
                                                    for d in [domain_reward_data, state_row, state_col]]

        # Forward pass in our neural net
        _, predictions = net(domain_reward_data, state_row, state_col,
                             args.num_vi, visualize=True)
        _, predicted_action = torch.max(
            predictions, dim=1, keepdim=True)
        predicted_action = predicted_action.item()
        r_move, c_move = GW.extract_action_direction(predicted_action)

        pred_traj[j + 1, 0] = current_state[0] + r_move
        pred_traj[j + 1, 1] = current_state[1] + c_move

        if pred_traj[j + 1, 0] == GW.target_x and pred_traj[j + 1, 1] == GW.target_y:
            pred_traj[j + 1:, 0] = GW.target_x
            pred_traj[j + 1:, 1] = GW.target_y
            break

    return pred_traj


def path_planning(net, args):
    # automatically select device to make the code device agnostic
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    correct, total = 0.0, 0.0
    for n_dom in range(args.dom_num):
        domain, goal, flag_map = create_map(args)
        if flag_map == 0:
            continue
        GW = GridWorld(domain, goal[0], goal[1])
        reward_prior = GW.target_get_reward_prior()

        # Sample random trajectories to our goal
        paths = search_path(GW, args.traj_num)
        for i in range(args.traj_num):
            if len(paths[i]) > 1:
                pred_traj = predict_trajectory(
                    net, GW, reward_prior, paths[i], device)

                # Plot optimal and predicted path (also start, end)
                if pred_traj[-1, 0] == goal[0] and pred_traj[-1, 1] == goal[1]:
                    correct += 1
                    if args.plot is True:
                        visualize_path(
                            GW.domain, paths[i], pred_traj, correct)
                        visualize_reward(
                            net.reward_image, correct)
                        visualize_v_value(
                            net.v_value_image, correct)
                total += 1
        sys.stdout.write("\r" + 'Progress: ' +
                         str(int((n_dom / args.dom_num) * 100)) + "%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    print('Rollout Accuracy: {:.2f}%'.format(100 * (correct / total)))


def set_args():
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='../data/vin_8x8.pth', help='Path to trained weights')
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

    # instantiate VIN net
    vin = VIN(args)

    # load net parameters
    vin.load_state_dict(torch.load(args.weights))

    # path planning
    path_planning(net=vin, args=args)
