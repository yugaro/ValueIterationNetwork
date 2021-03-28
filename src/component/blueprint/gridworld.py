import numpy as np
from collections import OrderedDict
np.random.seed(0)


class GridWorld:
    def __init__(self, domain, target_x, target_y):
        self.domain = domain
        self.n_row = domain.shape[0]
        self.n_col = domain.shape[1]
        self.obstacles = np.where(self.domain == 0)
        self.freespace = np.where(self.domain == 1)
        self.target_x = target_x
        self.target_y = target_y
        self.n_states = self.n_row * self.n_col
        self.ACTION = OrderedDict(N=(-1, 0), S=(1, 0), E=(0, 1), W=(0, -1),
                                  NE=(-1, 1), NW=(-1, -1), SE=(1, 1), SW=(1, -1))
        self.n_actions = len(self.ACTION)
        self.w_adj, self.non_obstacles, self.P, self.R = self.set_vals()

    def target_get_reward_prior(self):
        reward = np.zeros((self.n_row, self.n_col))
        reward[self.target_x, self.target_y] = 10
        return reward

    def loc_to_state(self, row, col):
        return np.ravel_multi_index([row, col], (self.n_row, self.n_col), order='C')

    def state_to_loc(self, state):
        return np.unravel_index(state, (self.n_row, self.n_col), order='C')

    def move(self, row, col, action):
        # Returns new [row,col] if we take the action
        r_move, c_move = self.ACTION[action]
        new_row = max(0, min(row + r_move, self.n_row - 1))
        new_col = max(0, min(col + c_move, self.n_col - 1))
        if self.domain[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col

    def set_vals(self):
        action_cost = np.linalg.norm(list(self.ACTION.values()), axis=1)
        R = - np.ones((self.n_states, self.n_actions)) * action_cost
        target = self.loc_to_state(self.target_x, self.target_y)
        R[target, :] = 0

        P = np.zeros((self.n_states, self.n_states, self.n_actions))
        for row in range(self.n_row):
            for col in range(self.n_col):
                if self.domain[row, col] == 1:
                    curr_state = self.loc_to_state(row, col)
                    for i_action, action in enumerate(self.ACTION):
                        neighbor_row, neighbor_col = self.move(
                            row, col, action)
                        neighbor_state = self.loc_to_state(
                            neighbor_row, neighbor_col)
                        if curr_state != neighbor_state:
                            P[curr_state, neighbor_state, i_action] = 1

        w_adj = np.maximum.reduce(P * action_cost, axis=2)

        non_obstacles = self.loc_to_state(self.freespace[0], self.freespace[1])
        non_obstacles = np.sort(non_obstacles)

        return w_adj, non_obstacles, P, R

    def extract_action_direction(self, action_num):
        for i_action, action in enumerate(self.ACTION):
            if i_action == action_num:
                r_move, c_move = self.ACTION[action]
                return r_move, c_move
