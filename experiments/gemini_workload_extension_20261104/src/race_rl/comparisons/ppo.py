"""PPO comparator only. No experimental superiority over the formal policy is established."""
import torch
from torch import nn
from ..policy import AutoregressivePolicy
from ..training import RACETrainer,seed_all


class ComparisonActorCritic(AutoregressivePolicy):
    def __init__(self,width=32,state_variant="full"):
        super().__init__(width,state_variant)
        self.critic = nn.Linear(width,1)

    def value(self,g):
        return self.critic(g).squeeze()


class PPOTrainer(RACETrainer):
    def __init__(self,*,seed=7,width=32,state_variant="full",lr=3e-4,clip=0.2,
                 epochs=4,entropy_coefficient=0.01,precision="float32"):
        seed_all(seed)
        if lr <= 0 or not 0 < clip < 1 or epochs < 1:
            raise ValueError("invalid PPO hyperparameters")
        if precision not in {"float32","float64"}:
            raise ValueError("unsupported policy precision")
        self.config = dict(algorithm="ppo_comparison",seed=seed,width=width,state_variant=state_variant,lr=lr,clip=clip,
                           epochs=epochs,entropy_coefficient=entropy_coefficient,gamma=1.0,precision=precision)
        self.policy = ComparisonActorCritic(width,state_variant).to(
            dtype=torch.float64 if precision == "float64" else torch.float32)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),lr=lr)
        self.episodes,self.training_evaluations,self.history = 0,0,[]


    def update(self,samples,stats):
        returns,total = [],0.0
        for _,_,_,_,r in reversed(samples):
            total = r+total  # full Monte Carlo episodic return, gamma=1
            returns.append(total)
        dtype = next(self.policy.parameters()).dtype
        returns = torch.tensor(list(reversed(returns)),dtype=dtype)
        old_log = torch.tensor([s[2] for s in samples],dtype=dtype)
        old_values = torch.tensor([s[3] for s in samples],dtype=dtype)
        advantages = returns-old_values
        if len(samples) > 1:
            advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
        for _ in range(self.config["epochs"]):
            logs,values,entropies = [],[],[]
            for o,index,*_ in samples:
                distribution,value = self.policy.distribution(o)
                logs.append(distribution.log_prob(torch.tensor(index)))
                values.append(value)
                entropies.append(distribution.entropy())
            ratio = torch.exp(torch.stack(logs)-old_log)
            clipped = torch.clamp(ratio,1-self.config["clip"],1+self.config["clip"])
            actor_loss = -torch.minimum(ratio*advantages,clipped*advantages).mean()
            value_loss = (torch.stack(values)-returns).square().mean()
            entropy = torch.stack(entropies).mean()
            loss = actor_loss+0.5*value_loss-self.config["entropy_coefficient"]*entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(),0.5)
            self.optimizer.step()
            stats["parameter_updates"] += 1
        stats.update(actor_loss=actor_loss.item(),value_loss=value_loss.item(),entropy=entropy.item())

