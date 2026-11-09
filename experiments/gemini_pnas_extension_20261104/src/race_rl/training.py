"""Formal REINFORCE policy gradient using only action-triggered environment rewards."""
import random
import time
import math
import numpy as np
import torch
from .policy import AutoregressivePolicy


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class RACETrainer:
    def __init__(self,*,seed=7,width=32,state_variant="full",lr=3e-4,
                 entropy_coefficient=0.0,precision="float32",policy_backend="design"):
        seed_all(seed)
        if lr <= 0 or entropy_coefficient < 0:
            raise ValueError("invalid policy-gradient hyperparameters")
        if precision not in {"float32","float64"}:
            raise ValueError("unsupported policy precision")
        self.config = dict(algorithm="reinforce",seed=seed,width=width,state_variant=state_variant,
            lr=lr,entropy_coefficient=entropy_coefficient,gamma=1.0,precision=precision)
        if policy_backend not in ('design','gemmini','gemmini_cnn'): raise ValueError('unknown policy backend')
        policy_class = AutoregressivePolicy
        if policy_backend == 'gemmini':
            from .accelerators.gemmini_policy import GemminiPolicy
            policy_class = GemminiPolicy
            self.config['policy_backend'] = policy_backend
        if policy_backend == 'gemmini_cnn':
            from .accelerators.cnn_policy import CNNPolicy
            policy_class = CNNPolicy
            self.config['policy_backend'] = policy_backend
        self.policy = policy_class(width,state_variant).to(
            dtype=torch.float64 if precision == "float64" else torch.float32)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),lr=lr)
        self.episodes,self.training_evaluations,self.history = 0,0,[]

    def run_episode(self,env,*,learn=True):
        observation = env.reset()
        samples = []
        timings = {'action_selection_time_s':0.0,'environment_step_time_s':0.0,'policy_update_time_s':0.0}
        def progress(stage):
            if not hasattr(env,'progress_path'):return
            from .serialization import save_json
            save_json(env.progress_path,{**getattr(env,'progress_context',{}),'stage':stage,'episode':self.episodes+1,
                'evaluations':env.oracle.evaluations,'evaluation_budget':env.oracle.budget,
                'requests':env.oracle.requests,'cache_hits':env.oracle.cache_hits,
                'current_ppa':({'power_w':env.current_record.average_power_w,'latency_s':env.current_record.latency_s,
                    'area_mm2':env.current_record.area_mm2} if env.current_record is not None else None),
                'last_reward':samples[-1][4] if samples else None,'best_objective':env.best_objective})
        while not env.done:
            progress('SELECTING_ACTION')
            started=time.perf_counter()
            action,log_probability,value = self.policy.select(observation)
            timings["action_selection_time_s"]+=time.perf_counter()-started
            index = action if getattr(self.policy,'structured_actions',False) else observation.legal_actions.index(action)
            progress("ENVIRONMENT_STEP")
            started=time.perf_counter()
            next_o,r,done = env.step(action)
            timings["environment_step_time_s"]+=time.perf_counter()-started
            if r is None or not math.isfinite(r):
                raise ValueError('environment returned unscorable feedback; missing PPA must not become a surrogate reward')
            samples.append((observation,index,log_probability,value,r))
            observation = next_o
            progress("RECEIVED_FEEDBACK")
        stats = {"episode":self.episodes,"steps":len(samples),"oracle_evaluations":env.oracle.evaluations,
                 "oracle_requests":env.oracle.requests,"cache_hits":env.oracle.cache_hits,
                 "best_objective":env.best_objective,"return":sum(s[4] for s in samples),
                 "termination":env.termination,"parameter_updates":0,
                 "algorithm":self.config["algorithm"],"critic_used":self.config["algorithm"]=="ppo_comparison"}
        support_size=lambda o:o.action_support_size if getattr(self.policy,'structured_actions',False) else len(o.legal_actions)
        supports = [support_size(s[0]) for s in samples]
        stats.update(initial_action_support=supports[0] if supports else support_size(observation),
            mean_action_support=float(np.mean(supports)) if supports else 0.0,
            max_action_support=max(supports,default=0))
        if learn and samples:
            progress("UPDATING_POLICY")
            started=time.perf_counter()
            self.update(samples,stats)
            timings["policy_update_time_s"]=time.perf_counter()-started
        if hasattr(env,"progress_path"):
            stats.update(timings)
            progress("EPISODE_COMPLETED")
        if learn:
            self.episodes += 1
            self.training_evaluations += env.oracle.evaluations
            self.history.append(stats)
        return stats

    def update(self,samples,stats):
        # REINFORCE reward-to-go, one update of the on-policy trajectory.
        # There is no critic, fitted baseline, surrogate ratio or policy clip.
        returns,total = [],0.0
        for _,_,_,_,r in reversed(samples):
            total += r
            returns.append(total)
        dtype = next(self.policy.parameters()).dtype
        returns = torch.tensor(list(reversed(returns)),dtype=dtype)
        logs,entropies = [],[]
        for o,index,*_ in samples:
            if getattr(self.policy,'structured_actions',False):
                logp,entropy=self.policy.evaluate_action(o,index)
                logs.append(logp);entropies.append(entropy)
            else:
                distribution,_ = self.policy.distribution(o)
                logs.append(distribution.log_prob(torch.tensor(index)))
                entropies.append(distribution.entropy())
        # Sum over steps: the episodic policy-gradient estimator.
        policy_loss = -(torch.stack(logs)*returns).sum()
        entropy = torch.stack(entropies).mean()
        loss = policy_loss-self.config['entropy_coefficient']*entropy
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm=torch.nn.utils.clip_grad_norm_(self.policy.parameters(),0.5)
        self.optimizer.step()
        stats.update(parameter_updates=1,policy_loss=policy_loss.item(),entropy=entropy.item(),
            normalized_entropy=float(torch.stack([e/math.log(s[0].action_support_size if getattr(self.policy,'structured_actions',False) else len(s[0].legal_actions))
                if (s[0].action_support_size if getattr(self.policy,'structured_actions',False) else len(s[0].legal_actions))>1 else e*0 for e,s in zip(entropies,samples)]).mean().detach()),
            gradient_norm_before_clip=float(gradient_norm),gradient_clip_limit=0.5,
            nonzero_reward_steps=sum(abs(s[4])>1e-12 for s in samples),
            first_step_reward_fraction=samples[0][4]/sum(s[4] for s in samples) if sum(s[4] for s in samples) else 0.0,
            algorithm='reinforce',critic_used=False)

    def save(self,path,*,experiment_identity):
        torch.save({"schema":2,"config":self.config,"policy":self.policy.state_dict(),
            "optimizer":self.optimizer.state_dict(),"episodes":self.episodes,"history":self.history,
            "training_evaluations":self.training_evaluations,"torch_rng":torch.get_rng_state(),
            "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),
            "experiment_identity":experiment_identity},path)

    @classmethod
    def load(cls,path,*,experiment_identity):
        # Only checkpoints generated locally by this project; pickle is never a remote input.
        saved = torch.load(path,map_location="cpu",weights_only=False)
        if saved["schema"] != 2 or saved["experiment_identity"] != experiment_identity:
            raise ValueError("checkpoint experiment identity mismatch")
        config = dict(saved["config"])
        config.pop("gamma")
        algorithm = config.pop("algorithm")
        expected = "reinforce" if cls is RACETrainer else "ppo_comparison"
        if algorithm != expected: raise ValueError("checkpoint learning algorithm mismatch")
        config.setdefault("precision","float32")  # historical checkpoints
        trainer = cls(**config)
        trainer.policy.load_state_dict(saved["policy"])
        trainer.optimizer.load_state_dict(saved["optimizer"])
        trainer.episodes,trainer.history = saved["episodes"],saved["history"]
        trainer.training_evaluations = saved["training_evaluations"]
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        return trainer
