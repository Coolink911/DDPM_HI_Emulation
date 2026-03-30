"""
Conditional Dataset loader with labels
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class ConditionalImageDataset(Dataset):
    def __init__(self, data_path, label_path, transform=None, label_stats=None):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        self.transform = transform
        self.label_stats = label_stats

        assert len(self.data) == len(self.labels), f"Data and labels length mismatch! {len(self.data)} vs {len(self.labels)}"

        print(f"Loaded {len(self.data)} images | Image shape: {self.data.shape[1:]} | Label shape: {self.labels.shape[1:]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx]).float()
        label = torch.from_numpy(self.labels[idx]).float()

        # Normalize image to [-1, 1]
        img = img * 2.0 - 1.0

        # Normalize labels
        if self.label_stats is not None:
            label = (label - self.label_stats['mean']) / self.label_stats['std']

        if img.dim() == 2:
            img = img.unsqueeze(0)

        return img, label


def get_conditional_dataloaders(
    data_dir='./data/params_2',
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    normalize_labels=True
):
    is_6param = 'params_6' in data_dir

    if is_6param:
        train_data = os.path.join(data_dir, 'train_LH_6.npy')
        val_data = os.path.join(data_dir, 'val_LH_6.npy')
        test_data = os.path.join(data_dir, 'test_LH_6.npy')
        train_labels = os.path.join(data_dir, 'train_labels_LH.npy')
        val_labels = os.path.join(data_dir, 'val_labels_LH.npy')
        test_labels = os.path.join(data_dir, 'test_labels_LH.npy')
    else:
        train_data = os.path.join(data_dir, 'train_LH.npy')
        val_data = os.path.join(data_dir, 'val_LH.npy')
        test_data = os.path.join(data_dir, 'test_LH.npy')
        train_labels = os.path.join(data_dir, 'train_labels_LH_2.npy')
        val_labels = os.path.join(data_dir, 'val_labels_LH_2.npy')
        test_labels = os.path.join(data_dir, 'test_labels_LH_2.npy')

    print(f"Loading dataset from {data_dir} ({'6-param' if is_6param else '2-param'})")

    # Label normalization stats
    label_stats = None
    if normalize_labels:
        train_labels_array = np.load(train_labels)
        label_mean = train_labels_array.mean(axis=0)
        label_std = train_labels_array.std(axis=0)
        label_stats = {'mean': torch.from_numpy(label_mean).float(), 'std': torch.from_numpy(label_std).float()}
        print(f"Label normalization → mean={label_mean}, std={label_std}")

    train_dataset = ConditionalImageDataset(train_data, train_labels, label_stats=label_stats)
    val_dataset   = ConditionalImageDataset(val_data,   val_labels,   label_stats=label_stats)
    test_dataset  = ConditionalImageDataset(test_data,  test_labels,  label_stats=label_stats)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    return train_loader, val_loader, test_loader
