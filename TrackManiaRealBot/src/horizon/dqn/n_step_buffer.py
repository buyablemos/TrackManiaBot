import numpy as np
import torch

from ...config import Config
from ...utils.utils import from_schedule

class NStepBuffer:
    def __init__(self, n_steps: int, device) -> None:
        self.n_steps = n_steps
        self.current_size = 0
        self.position = 0
        self.device = device

        self.states = torch.zeros((self.n_steps, Config.Arch.INPUT_SIZE), dtype=torch.float, device=self.device)
        self.actions = torch.zeros(self.n_steps, dtype=torch.int64, device=self.device)
        self.rewards = torch.zeros(self.n_steps, dtype=torch.float, device=self.device)
        self.action_matches_argmax = torch.zeros(self.n_steps, dtype=torch.bool, device=self.device)
        self.gammas = torch.tensor(from_schedule(Config.DQN.GAMMA_SCHEDULE, 0) ** np.arange(self.n_steps), dtype=torch.float, device=self.device)

    def __len__(self):
        return self.current_size

    def clear(self):
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.action_matches_argmax.zero_()
        self.current_size = 0
        self.position = 0

    def get_transition(self, time):
        with torch.no_grad():
            self.gammas.copy_(torch.tensor(from_schedule(Config.DQN.GAMMA_SCHEDULE, time) ** np.arange(self.n_steps), dtype=torch.float, device=self.device))
        idx = (self.position - self.current_size) % self.n_steps
        return self.states[idx], self.actions[idx], self.cumulative_reward()

    def pop_transition(self):
        if self.current_size > 0:
            self.current_size -= 1

    def is_full(self):
        return self.current_size == self.n_steps

    def is_empty(self):
        return self.current_size == 0

    def cumulative_reward(self):
        # Create indices array directly
        indices = torch.arange(self.position - self.current_size, self.position, device=self.device) % self.n_steps
        
        # Find first False in action_matches_argmax using torch operations
        matches = self.action_matches_argmax[indices]
        cutoff_idx = self.current_size
        if not matches.all():
            cutoff_idx = torch.where(~matches)[0][0].item()
                
        # Only use rewards up to cutoff
        rewards_tensor = self.rewards[indices[:cutoff_idx]].to(self.device)
        gammas_tensor = self.gammas[:cutoff_idx].to(self.device)
        return torch.sum(rewards_tensor * gammas_tensor)

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, action_matches_argmax: bool):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.action_matches_argmax[self.position] = action_matches_argmax

        self.position = (self.position + 1) % self.n_steps
        if self.current_size < self.n_steps:
            self.current_size += 1
