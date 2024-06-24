import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, observations, actions):
        # observations = nt x  traj_length x (1x4x84x84) (4 stacked frames)
        # actions = nt x  traj_length x (1) (action)
        self.observations = [frame for traj in observations for frame in traj]
        self.actions = [action for traj in actions for action in traj]

    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, index):
        observation = self.observations[index]
        action = self.actions[index]
        return observation, action