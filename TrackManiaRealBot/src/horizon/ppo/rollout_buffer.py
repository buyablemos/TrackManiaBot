import torch
from torch import Tensor

from src.config import Config

class RolloutBuffer:
    def __init__(self, device):
        self.device = device

        self.states = torch.zeros((Config.PPO.MEMORY_SIZE, Config.Arch.INPUT_SIZE), dtype=torch.float, device=self.device)
        self.actions = torch.zeros(Config.PPO.MEMORY_SIZE, dtype=torch.int64, device=self.device)
        self.probs = torch.zeros(Config.PPO.MEMORY_SIZE, dtype=torch.float, device=self.device)
        self.rewards = torch.zeros(Config.PPO.MEMORY_SIZE, dtype=torch.float, device=self.device)
        self.dones = torch.zeros(Config.PPO.MEMORY_SIZE, dtype=torch.int, device=self.device)
        self.values = torch.zeros(Config.PPO.MEMORY_SIZE, dtype=torch.float, device=self.device)

        self.batch_size = Config.PPO.BATCH_SIZE
        self.position = 0

    def is_full(self):
        return self.position >= Config.PPO.MEMORY_SIZE

    def clear(self):
        self.states.zero_()
        self.actions.zero_()
        self.probs.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.values.zero_()

        self.position = 0

    def add(self, state, action, log_prob, reward, done, value):
        assert self.position < Config.PPO.MEMORY_SIZE, "Rollout buffer is full. Please clear it before adding new data."

        self.states[self.position] = state
        self.actions[self.position] = action
        self.probs[self.position] = log_prob.detach()
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.values[self.position] = value.detach()
        self.position += 1

    def generate_batches(self) -> Tensor:
        assert self.is_full(), "Rollout buffer is not full. Please fill it before generating batches."

        num_batches = self.position // self.batch_size

        indices = torch.randperm(self.position, device=self.device)  # Randomly permute the indices of the buffer
        batches = indices.view(num_batches, self.batch_size) # batches is a tensor of arrays of indices. These indices indicate a batch of states, actions, probs, values, rewards and dones
        return batches

    def get_buffer(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        assert self.is_full()
        return self.states, self.actions, self.probs, self.values, self.rewards, self.dones