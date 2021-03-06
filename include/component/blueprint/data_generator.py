import numpy as np
import torch
import torch.utils.data as data
np.random.seed(0)


class GridWorldData(data.Dataset):
    def __init__(self, file, dom_size, train=True, transform=None, target_transform=None):
        assert file.endswith('.npz')
        self.file = file
        self.dom_size = dom_size
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.images, self.S1, self.S2, self.labels = self._process(
            file, self.train)

    def __getitem__(self, index):
        img = self.images[index]
        s1 = self.S1[index]
        s2 = self.S2[index]
        label = self.labels[index]

        # Apply transform if we have one
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)

        # Apply target transform if we have one
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, int(s1), int(s2), int(label)

    def __len__(self):
        return self.images.shape[0]

    def _process(self, file, train):
        with np.load(file, mmap_mode='r') as f:
            if train:
                images = f['arr_0']
                S1 = f['arr_1']
                S2 = f['arr_2']
                labels = f['arr_3']
            else:
                images = f['arr_4']
                S1 = f['arr_5']
                S2 = f['arr_6']
                labels = f['arr_7']

        # Set proper datatypes
        images = images.astype(np.float32)
        S1 = S1.astype(int)
        S2 = S2.astype(int)
        labels = labels.astype(int)

        # Print number of samples
        if train:
            print("Number of Train Samples: {0}".format(images.shape[0]))
        else:
            print("Number of Test Samples: {0}".format(images.shape[0]))
        return images, S1, S2, labels
