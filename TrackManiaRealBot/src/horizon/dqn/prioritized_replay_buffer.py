import torch
from src.config import Config

class PrioritizedReplayBuffer:

    def __init__(self, capacity, alpha, beta, device):
        self.capacity = capacity
        self.device = device

        self.states = torch.zeros((capacity, Config.Arch.INPUT_SIZE), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, Config.Arch.INPUT_SIZE), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)

        self.priorities = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        self.pos = 0
        self.fill_level = 0

        self.alpha = alpha
        self.beta = beta
        self.epsilon = 0.001

    def __len__(self):
        return self.fill_level

    def add(self, transition, priority=None):
        state, action, reward, next_state, done = transition
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done

        if priority is None:
            priority = self.priorities[:self.fill_level].max().item() if self.fill_level > 0 else 1.0

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        self.fill_level = min(self.fill_level + 1, self.capacity)

    def sample(self, batch_size):
        if self.fill_level < Config.DQN.MIN_MEMORY or batch_size > self.fill_level:
            return None

        self.beta += (Config.DQN.BETA_MAX - Config.DQN.BETA_START) / Config.DQN.BETA_INCREMENT_STEPS
        self.beta = min(self.beta, Config.DQN.BETA_MAX)

        valid_priorities = self.priorities[:self.fill_level]
        scaled = (valid_priorities + self.epsilon) ** self.alpha
        probabilities = scaled / scaled.sum()

        indices = torch.multinomial(probabilities, batch_size, replacement=False)

        sampled_states = self.states[indices]
        sampled_actions = self.actions[indices]
        sampled_rewards = self.rewards[indices]
        sampled_next_states = self.next_states[indices]
        sampled_dones = self.dones[indices]

        weights = ((1.0 / self.fill_level) * (1.0 / probabilities[indices])) ** self.beta
        weights /= weights.max()

        return (sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones), indices, weights

    def update_priorities(self, indices, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach()
        self.priorities[indices] = priorities


