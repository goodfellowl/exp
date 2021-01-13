
""" Sampler for dataloader. """
import torch
import numpy as np

class CategoriesSampler():
    """The class to generate episodic data
    label : dataset label
    n_batch: the number of batch of episodic data
    n_cls:  the class number of batch
    n_per:  the picture number of batch """


    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
        
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                length = self.m_ind[c]
                pos = torch.randperm(len(length))[:self.n_per]
                batch.append(length[pos])
            batch = torch.stack(batch).reshape(-1)
            yield batch

# class CategoriesSampler():

#     def __init__(self, label, n_batch, n_cls, n_per, ep_per_batch=1):
#         self.n_batch = n_batch
#         self.n_cls = n_cls
#         self.n_per = n_per
#         self.ep_per_batch = ep_per_batch

#         label = np.array(label)
#         self.catlocs = []
#         for c in range(max(label) + 1):
#             self.catlocs.append(np.argwhere(label == c).reshape(-1))

#     def __len__(self):
#         return self.n_batch
    
#     def __iter__(self):
#         for i_batch in range(self.n_batch):
#             batch = []
#             for i_ep in range(self.ep_per_batch):
#                 episode = []
#                 classes = np.random.choice(len(self.catlocs), self.n_cls,
#                                            replace=False)
#                 for c in classes:
#                     l = np.random.choice(self.catlocs[c], self.n_per,
#                                          replace=False)
#                     episode.append(torch.from_numpy(l))
#                 episode = torch.stack(episode)
#                 batch.append(episode)
#             batch = torch.stack(batch) # bs * n_cls * n_per
#             yield batch.view(-1)
