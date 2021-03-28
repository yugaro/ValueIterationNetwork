import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
np.random.seed(0)


def get_path_row(start, goal, predecessors):
    path = []
    i = goal
    while i != start and i >= 0:
        path.append(i)
        i = predecessors[i]
    if i < 0:
        return []
    path.append(i)
    return path[::-1]


def get_path(start, goal, predecessors):
    return get_path_row(start, goal, predecessors[start])


def search_path(GW, traj_num):
    # sample trajectories from random nodes in the domain.
    w_adj = GW.w_adj
    csr = csr_matrix(w_adj)
    goal_loc = GW.loc_to_state(GW.target_x, GW.target_y)
    _, predecessors = shortest_path(csr, return_predecessors=True)

    paths = []
    i = 0
    while i < traj_num:
        start_loc = np.random.choice(GW.non_obstacles)
        path_loc = get_path(start=start_loc, goal=goal_loc,
                            predecessors=predecessors)
        if not path_loc:
            continue
        else:
            states = []
            for j in range(len(path_loc)):
                state = GW.state_to_loc(path_loc[j])
                states.append(list(state))
            i += 1
        paths.append(np.array(states))
    return paths
