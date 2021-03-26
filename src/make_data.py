import sys
import numpy as np
import argparse
from components.gridworld import GridWorld
from components.sample_trajectory import sample_trajectory
from components.obstacles import obstacles
sys.path.append('.')
sys.path.remove('.')


def extract_action(states_xy):
    # Outputs a 1D vector of actions corresponding to the state.
    n_actions = 8
    action_vecs = np.asarray([[-1., 0.], [1., 0.], [0., 1.], [0., -1.],
                              [-1., 1.], [-1., -1.], [1., 1.], [1., -1.]])
    action_vecs[4:] = 1 / np.sqrt(2) * action_vecs[4:]
    action_vecs = action_vecs.T
    state_diff = np.diff(states_xy, axis=0)
    norm_state_diff = state_diff * np.tile(
        1 / np.sqrt(np.sum(np.square(state_diff), axis=1)), (2, 1)).T
    prj_state_diff = np.dot(norm_state_diff, action_vecs)
    actions_one_hot = np.abs(prj_state_diff - 1) < 0.00001
    actions = np.dot(actions_one_hot, np.arange(n_actions).T)
    return actions


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


def make_data(dom_size, dom_num, max_obs_num, max_obs_size, traj_num,
              state_batch_size):
    X_l = []
    S1_l = []
    S2_l = []
    Labels_l = []
    n_dom = 0
    while n_dom <= dom_num:
        im, goal, flag_map = create_map(args)
        if flag_map == 0:
            continue
        # generate gridworld from obstacle map
        G = GridWorld(im, goal[0], goal[1])
        # get value prior
        value_prior = G.target_get_reward_prior()

        # sample random trajectories to our goal
        states_xy, states_one_hot = sample_trajectory(G, traj_num)
        for i in range(traj_num):
            if len(states_xy[i]) > 1:
                # invert domain image => 0 = free, 1 = obstacle
                image = 1 - im

                # resize domain and goal images and concate
                ns = states_xy[i].shape[0] - 1
                image_data = np.resize(image, (1, 1, dom_size[0], dom_size[1]))
                value_data = np.resize(value_prior,
                                       (1, 1, dom_size[0], dom_size[1]))
                iv_mixed = np.concatenate((image_data, value_data), axis=1)
                X_current = np.tile(iv_mixed, (ns, 1, 1, 1))

                # resize states
                S1_current = np.expand_dims(states_xy[i][0:ns, 0], axis=1)
                S2_current = np.expand_dims(states_xy[i][0:ns, 1], axis=1)

                # get optimal actions for each state
                actions = extract_action(states_xy[i])

                # resize labels
                Labels_current = np.expand_dims(actions, axis=1)

                # append to output list
                X_l.append(X_current)
                S1_l.append(S1_current)
                S2_l.append(S2_current)
                Labels_l.append(Labels_current)
        n_dom += 1
        sys.stdout.write("\r" + 'Progress: ' + str(int((n_dom / dom_num) * 100)) + "%")
        sys.stdout.flush()
    sys.stdout.write('\n')

    # concat all outputs
    X_f = np.concatenate(X_l)
    S1_f = np.concatenate(S1_l)
    S2_f = np.concatenate(S2_l)
    Labels_f = np.concatenate(Labels_l)
    return X_f, S1_f, S2_f, Labels_f


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dom_size", "-s", type=int,
                        help="size of domain", default=8)
    parser.add_argument("--dom_num", "-nd", type=int,
                        help="number of domains", default=5000)
    parser.add_argument("--max_obs_num", "-no", type=int,
                        help="maximum number of obstacles", default=50)
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

    # set path to save dataset
    save_path = 'data/gridworld_{0}x{1}'.format(args.dom_size, args.dom_size)

    # make training data
    print("Now making training data...")
    X_out_tr, S1_out_tr, S2_out_tr, Labels_out_tr = make_data(
        dom_size=(args.dom_size, args.dom_size),
        dom_num=args.dom_num,
        max_obs_num=args.max_obs_num,
        max_obs_size=args.max_obs_size,
        traj_num=args.traj_num,
        state_batch_size=args.state_batch_size)

    # make testing data
    print("\nNow making  testing data...")
    X_out_ts, S1_out_ts, S2_out_ts, Labels_out_ts = make_data(
        dom_size=(args.dom_size, args.dom_size),
        dom_num=args.dom_num / 6,
        max_obs_num=args.max_obs_num,
        max_obs_size=args.max_obs_size,
        traj_num=args.traj_num,
        state_batch_size=args.state_batch_size)

    # save dataset
    np.savez_compressed(save_path, X_out_tr, S1_out_tr, S2_out_tr,
                        Labels_out_tr, X_out_ts, S1_out_ts, S2_out_ts,
                        Labels_out_ts)
