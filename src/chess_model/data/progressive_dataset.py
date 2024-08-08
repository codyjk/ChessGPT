import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class ProgressiveDataset(Dataset):
    """
    Wrapper around a Dataset that progressively increases the size of the
    dataset by a factor of growth_factor, starting from initial_size_fraction
    of the original size.

    This allows for a more efficient training process, as the dataset is
    progressively grown as the model is trained. It also allows us to quickly
    evaluate performance of the model in early training epochs, as the initial
    dataset is small.
    """

    def __init__(self, base_dataset, initial_size_fraction=0.1, growth_factor=2):
        self.base_dataset = base_dataset
        self.initial_size_fraction = initial_size_fraction
        self.growth_factor = growth_factor
        self.current_size = int(len(base_dataset) * initial_size_fraction)
        self.indices = self.generate_indices()

    def generate_indices(self):
        return np.random.choice(
            len(self.base_dataset), self.current_size, replace=False
        )

    def grow(self):
        new_size = min(
            int(self.current_size * self.growth_factor), len(self.base_dataset)
        )
        if new_size > self.current_size:
            self.current_size = new_size
            self.indices = self.generate_indices()
            return True
        return False

    def __len__(self):
        return self.current_size

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]
