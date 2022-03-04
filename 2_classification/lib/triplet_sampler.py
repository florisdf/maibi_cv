import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import random


class TripletSampler(Sampler):
    """
    Return batches with anchors, positives and negatives.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        if not batch_size % 3 == 0:
            raise ValueError(
                'Batch size should be divisible by 3.'
            )

        _, sample_labels = zip(*dataset.samples)
        self.sample_labels = np.array(sample_labels)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.sample_labels) // (self.batch_size // 3)

    def __iter__(self):
        sample_idxs = list(range(len(self.sample_labels)))

        if self.shuffle:
            random.shuffle(sample_idxs)

        batch = []
        for anchor_idx in sample_idxs:
            anchor_label = self.sample_labels[anchor_idx]

            # Find sample indices with same label as anchor
            pos_idxs = np.where(self.sample_labels == anchor_label)[0]
            # Drop the anchor sample index itself from the positives
            pos_idxs = pos_idxs[pos_idxs != anchor_idx]

            if len(pos_idxs) == 0:
                continue

            neg_idxs = list(np.where(self.sample_labels != anchor_label)[0])

            pos_idx = np.random.choice(pos_idxs, 1)[0]
            neg_idx = np.random.choice(neg_idxs, 1)[0]

            batch.extend([anchor_idx, pos_idx, neg_idx])

            if len(batch) == self.batch_size:
                yield batch
                batch = []


def split_triplet_tensor(tensor):
    a_tensor =  tensor[0::3]
    p_tensor =  tensor[1::3]
    n_tensor =  tensor[2::3]

    return (
        a_tensor, p_tensor, n_tensor
    )