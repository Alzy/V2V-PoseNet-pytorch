import os
import numpy as np
import sys
import struct
from torch.utils.data import Dataset


class HEATHandDataset(Dataset):
    def __init__(self, voxel_dir, joint_dir, transform=None):
        """
        Args:
            voxel_dir (string): Directory with all the voxel grid files.
            joint_dir (string): Directory with all the joint location files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.voxel_dir = voxel_dir
        self.joint_dir = joint_dir
        self.transform = transform
        self.voxel_files = sorted([f for f in os.listdir(voxel_dir) if f.endswith('.npy')])
        self.joint_files = sorted([f for f in os.listdir(joint_dir) if f.endswith('.npy')])

        assert len(self.voxel_files) == len(self.joint_files), "Mismatch between voxel and joint files count"

    def __len__(self):
        return len(self.voxel_files)

    def __getitem__(self, idx):
        voxel_path = os.path.join(self.voxel_dir, self.voxel_files[idx])
        joint_path = os.path.join(self.joint_dir, self.joint_files[idx])

        voxel_grid = np.load(voxel_path)
        joint_locations = np.load(joint_path)

        sample = {'voxel_grid': voxel_grid, 'joints': joint_locations}

        if self.transform:
            sample = self.transform(sample)

        return sample