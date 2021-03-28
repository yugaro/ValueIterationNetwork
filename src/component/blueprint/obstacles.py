import numpy as np
np.random.seed(0)


class obstacles:
    def __init__(self, domsize=None, goal=None, size_max=None,
                 dom=None, obs_types=None, num_types=None):
        self.domsize = domsize or []
        self.goal = goal or []
        self.dom = dom or np.ones(self.domsize)
        self.obs_types = obs_types or ["circ", "rect"]
        self.num_types = num_types or len(self.obs_types)
        self.size_max = size_max or np.max(self.domsize) / 4

    def check_goal(self, dom=None):
        return dom[self.goal[0], self.goal[1]] == 0

    def insert_rect(self, x, y, height, width):
        # Insert a rectangular obstacle into map
        im_try = np.copy(self.dom)
        im_try[x:x + height, y:y + width] = 0
        return im_try

    def add_rand_obs(self, obj_type):
        # Add random (valid) obstacle to map
        if obj_type == "circ":
            print("circ is not yet implemented... sorry")
        elif obj_type == "rect":
            rand_height = int(np.ceil(np.random.rand() * self.size_max))
            rand_width = int(np.ceil(np.random.rand() * self.size_max))
            randx = int(np.ceil(np.random.rand() * (self.domsize[1] - 1)))
            randy = int(np.ceil(np.random.rand() * (self.domsize[1] - 1)))
            im_try = self.insert_rect(randx, randy, rand_height, rand_width)
        if self.check_goal(im_try):
            return False
        else:
            self.dom = im_try
            return True

    def add_n_rand_obs(self, n):
        # Add random (valid) obstacles to map
        count = 0
        for i in range(n):
            obj_type = "rect"
            if self.add_rand_obs(obj_type):
                count += 1
        return count

    def add_border(self):
        # Make full outer border an obstacle
        im_try = np.copy(self.dom)
        im_try[0:self.domsize[0], 0] = 0
        im_try[0, 0:self.domsize[1]] = 0
        im_try[0:self.domsize[0], self.domsize[1] - 1] = 0
        im_try[self.domsize[0] - 1, 0:self.domsize[1]] = 0
        if self.check_goal(im_try):
            return False
        else:
            self.dom = im_try
            return True
