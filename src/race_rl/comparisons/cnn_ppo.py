"""PPO comparison on the project's CNN actions, not a Chiplet-Gym reproduction."""
import math

import torch
from torch import nn

from ..accelerators.cnn_policy import CNNPolicy
from ..observation import TILE_FEATURE_SIZE
from ..training import RACETrainer, seed_all


class CNNActorCritic(CNNPolicy):
    def __init__(self, width=32, state_variant='full'):
        super().__init__(width, state_variant)
        self.critic = nn.Linear(width, 1)

    def value(self, observation):
        _, _, features = self.encoder(observation.graph)
        return self.critic(features).squeeze()

    def select(self, observation):
        from ..accelerators.cnn_design import transaction
        with torch.no_grad():
            choices, logp, _ = self._decisions(observation)
            action = transaction(observation.workload, observation.current_design,
                                 choices, observation.max_scope)
            return action, float(logp), float(self.value(observation))


class CNNPPOTrainer(RACETrainer):
    def __init__(self, *, seed=7, width=32, state_variant='full', lr=3e-4,
                 clip=0.2, epochs=4, entropy_coefficient=0.01, precision='float64',
                 gamma=1.0, gae_lambda=1.0):
        seed_all(seed)
        if (not math.isfinite(lr) or lr <= 0 or not 0 < clip < 1 or epochs < 1 or
                not math.isfinite(entropy_coefficient) or entropy_coefficient < 0 or
                precision not in ('float32', 'float64') or not 0 <= gamma <= 1 or
                not 0 <= gae_lambda <= 1):
            raise ValueError('invalid CNN PPO configuration')
        self.config = {'algorithm':'ppo_comparison', 'seed':seed, 'width':width,
            'state_variant':state_variant, 'lr':lr, 'clip':clip, 'epochs':epochs,
            'entropy_coefficient':entropy_coefficient, 'gamma':gamma,
            'gae_lambda':gae_lambda, 'precision':precision}
        self.policy = CNNActorCritic(width, state_variant).to(
            dtype=torch.float64 if precision == 'float64' else torch.float32)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.episodes = self.training_evaluations = 0
        self.history = []

    def update(self, samples, stats):
        gae = 0.0
        advantages_list = []
        for i in reversed(range(len(samples))):
            next_value = samples[i+1][3] if i+1 < len(samples) else 0.0
            delta = samples[i][4] + self.config['gamma']*next_value - samples[i][3]
            gae = delta + self.config['gamma']*self.config['gae_lambda']*gae
            advantages_list.append(gae)
        dtype = next(self.policy.parameters()).dtype
        advantages = torch.tensor(list(reversed(advantages_list)), dtype=dtype)
        returns = advantages + torch.tensor([sample[3] for sample in samples], dtype=dtype)
        old_log = torch.tensor([sample[2] for sample in samples], dtype=dtype)
        if len(samples) > 1:
            advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
        for _ in range(self.config['epochs']):
            logs, entropies, values = [], [], []
            for observation, action, *_ in samples:
                logp, entropy = self.policy.evaluate_action(observation, action)
                logs.append(logp)
                entropies.append(entropy)
                values.append(self.policy.value(observation))
            ratio = torch.exp(torch.stack(logs)-old_log)
            clipped = torch.clamp(ratio, 1-self.config['clip'], 1+self.config['clip'])
            actor_loss = -torch.minimum(ratio*advantages, clipped*advantages).mean()
            value_loss = (torch.stack(values)-returns).square().mean()
            entropy = torch.stack(entropies).mean()
            loss = actor_loss + 0.5*value_loss - self.config['entropy_coefficient']*entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            stats['parameter_updates'] += 1
        stats.update(actor_loss=float(actor_loss.detach()), value_loss=float(value_loss.detach()),
                     entropy=float(entropy.detach()), critic_used=True,
                     algorithm='ppo_comparison')


class FlatCNNEncoder(nn.Module):
    """Ten global/task features, with local projections for variable-size actions.

    This has no graph message passing. Local projections are necessary because
    this project's anchor/target decisions have no counterpart in Chiplet-Gym.
    """
    def __init__(self, width):
        super().__init__()
        self.global_net = nn.Sequential(nn.Linear(10,64), nn.Tanh(), nn.Linear(64,64),
                                        nn.Tanh(), nn.Linear(64,width), nn.Tanh())
        self.tile = nn.Sequential(nn.Linear(TILE_FEATURE_SIZE+25,width), nn.Tanh())
        self.hardware = nn.Sequential(nn.Linear(9,width), nn.Tanh())

    def forward(self, graph):
        dtype = self.global_net[0].weight.dtype
        features = (*graph.global_features[:4], *graph.task_features[:3], *graph.task_features[9:12])
        global_embedding = self.global_net(torch.tensor(features, dtype=dtype))
        return (self.tile(torch.tensor(graph.tile_features, dtype=dtype)),
                self.hardware(torch.tensor(graph.hardware_features, dtype=dtype)), global_embedding)


class ChipletGymAdaptedTrainer(CNNPPOTrainer):
    """Chiplet-Gym-inspired PPO adapted to CNN transactions; not its original optimizer."""
    def __init__(self, *, seed=7, width=32, epochs=10, precision='float64'):
        super().__init__(seed=seed, width=width, lr=3e-4, clip=0.2, epochs=epochs,
                         entropy_coefficient=0.1, gamma=0.99, gae_lambda=0.95,
                         precision=precision)
        seed_all(seed)
        self.policy = CNNActorCritic(width)
        self.policy.encoder = FlatCNNEncoder(width)
        self.policy = self.policy.to(dtype=torch.float64 if precision == 'float64' else torch.float32)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.config.update(baseline='chiplet_gym_ppo_adapted',
            original_episode_length=2, original_rollout_steps=2048, original_batch_size=64,
            deviations='CNN action heads, flat 10-feature proxy, short on-policy episodes, shared PPA scorer; no original ensemble/SA')
