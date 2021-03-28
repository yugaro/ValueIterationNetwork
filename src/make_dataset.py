import argparse
import numpy as np
import sys
from component.blueprint.obstacles import obstacles
from component.blueprint.gridworld import GridWorld
from component.controller.search_path import search_path
np.random.seed(1)


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
    if n_obs == 0 or not border_res:
        flag_map = 0

    return obs.dom, goal, flag_map


def extract_action(states_xy):
    # Outputs a 1D vector of actions corresponding to the state.
    n_actions = 8
    action_vecs = np.array([[-1., 0.], [1., 0.], [0., 1.],
                            [0., -1.], [-1., 1.], [-1., -1.], [1., 1.], [1., -1.]])
    action_vecs[4:] = 1 / np.sqrt(2) * action_vecs[4:]
    action_vecs = action_vecs.T
    state_diff = np.diff(states_xy, axis=0)
    norm_state_diff = state_diff * np.tile(
        1 / np.sqrt(np.sum(np.square(state_diff), axis=1)), (2, 1)).T
    prj_state_diff = np.dot(norm_state_diff, action_vecs)
    actions_one_hot = np.abs(prj_state_diff - 1) < 0.00001
    actions = np.dot(actions_one_hot, np.arange(n_actions).T)

    return actions


def make_data(dom_size, dom_num, max_obs_num, max_obs_size, traj_num, state_batch_size):
    DRM = []
    SR = []
    SC = []
    AL = []

    for k in range(int(dom_num)):
        domain, goal, flag_map = create_map(args)
        if flag_map == 0:
            continue
        GW = GridWorld(domain, goal[0], goal[1])
        reward_prior = GW.target_get_reward_prior()

        # sample random trajectories to our goal
        paths = search_path(GW, traj_num)
        for i in range(len(paths)):
            if len(paths[i]) > 1:
                # resize domain and goal images and concate
                num_states = len(paths[i]) - 1
                domain_data = domain.reshape(1, 1, dom_size[0], dom_size[1])
                reward_data = reward_prior.reshape(
                    1, 1, dom_size[0], dom_size[1])
                domain_reward_data = np.concatenate(
                    (domain_data, reward_data), axis=1)
                domain_reward_map = np.tile(
                    domain_reward_data, (num_states, 1, 1, 1))

                state_row = np.expand_dims(paths[i][:num_states, 0], axis=1)
                state_column = np.expand_dims(paths[i][:num_states, 1], axis=1)

                action_data = extract_action(paths[i])
                action_label = np.expand_dims(action_data, axis=1)

                DRM.append(domain_reward_map)
                SR.append(state_row)
                SC.append(state_column)
                AL.append(action_label)
        sys.stdout.write("\r" + 'Progress: ' +
                         str(int((k / dom_num) * 100)) + "%")
        sys.stdout.flush()
    sys.stdout.write('\n')

    # concat all outputs
    DRMf = np.concatenate(DRM)
    SRf = np.concatenate(SR)
    SCf = np.concatenate(SC)
    ALf = np.concatenate(AL)
    return DRMf, SRf, SCf, ALf


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dom_size", "-s", type=int,
                        help="size of domain", default=8)
    parser.add_argument("--dom_num", "-nd", type=int,
                        help="number of domains", default=5000)
    parser.add_argument("--max_obs_num", "-no", type=int,
                        help="maximum number of obstacles", default=20)
    parser.add_argument("--max_obs_size", "-os", type=int,
                        help="maximum obstacle size", default=2)
    parser.add_argument("--traj_num", "-nt", type=int,
                        help="number of trajectories", default=7)
    parser.add_argument("--state_batch_size", "-bs", type=int,
                        help="state batch size", default=1)
    return parser.parse_args()

if __name__ == '__main__':
    # set args
    args = set_args()

    print("Now making training data...")
    DRMf_tr, SRf_tr, SCf_tr, ALf_tr = make_data(
        dom_size=(args.dom_size, args.dom_size),
        dom_num=args.dom_num,
        max_obs_num=args.max_obs_num,
        max_obs_size=args.max_obs_size,
        traj_num=args.traj_num,
        state_batch_size=args.state_batch_size)

    # make testing data
    print("\nNow making  testing data...")
    DRMf_ts, SRf_ts, SCf_ts, ALf_ts = make_data(
        dom_size=(args.dom_size, args.dom_size),
        dom_num=args.dom_num / 6,
        max_obs_num=args.max_obs_num,
        max_obs_size=args.max_obs_size,
        traj_num=args.traj_num,
        state_batch_size=args.state_batch_size)

    # save dataset
    save_path = '../data/gridworld_{0}x{1}'.format(args.dom_size, args.dom_size)
    np.savez_compressed(save_path, DRMf_tr, SRf_tr, SCf_tr, ALf_tr,
                        DRMf_ts, SRf_ts, SCf_ts, ALf_ts)
