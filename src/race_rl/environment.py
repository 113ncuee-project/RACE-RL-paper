"""Finite budget mapping episodes with a separately maintained verified archive."""
from .actions import apply, legal_actions
from .observation import observe
from .oracle import ActionOracle
from .rewards import potential, reward
from .serialization import digest
from .requirements import PPARequirements
from dataclasses import replace
import time


class RACEEnvironment:
    def __init__(self,workload,hardware,initial,*,budget=16,max_scope=4,max_requests=None,
                 reward_variant="dense",lambda_b=1.0,lambda_c=0.0,limits=None,scheduling_policy="shared_reads",requirements=None):
        self.requirements = requirements or PPARequirements(
            max_power_w=(limits or {}).get("average_power_w"),
            max_latency_s=(limits or {}).get("latency_s"),max_area_mm2=(limits or {}).get("area_mm2"))
        self.workload,self.hardware,self.initial = workload,hardware,initial
        self.settings = dict(budget=budget,max_scope=max_scope,max_requests=max_requests or budget,
            reward_variant=reward_variant,lambda_b=lambda_b,lambda_c=lambda_c,limits=limits or {},
            scheduling_policy=scheduling_policy)
        if self.settings["max_requests"] < 1 or max_scope < 1:
            raise ValueError("positive limits required")
        if reward_variant not in {"dense","terminal","potential"}:
            raise ValueError("unknown reward variant")
        if not set(self.settings["limits"]) <= {"energy_j","latency_s","area_mm2","average_power_w"}:
            raise ValueError("unknown strict PPA limit")
        if any(v <= 0 for v in self.settings["limits"].values()):
            raise ValueError("strict limits must be positive")

    def feasible(self,record):
        return self.scorer.assess(record)["feasible"] and all(getattr(record,k) <= v for k,v in self.settings["limits"].items())

    def reset(self):
        self.oracle = ActionOracle(self.workload,self.hardware,budget=self.settings["budget"],
                             scheduling_policy=self.settings["scheduling_policy"])
        self.current_mapping,self.best_mapping = self.initial,None
        self.current_record = self.best_record = None
        self.scorer = self.requirements.bind()
        self.reference = 1.0
        self.last_reward = 0.0
        self.initial_objective = self.current_objective = self.best_objective = 1.0
        self.trace = []
        self.structural_validation_time_s = 0.0
        self._refresh()
        return self.observation

    def _refresh(self):
        started = time.perf_counter()
        self.actions = legal_actions(self.workload,self.hardware,self.current_mapping,self.settings["max_scope"])
        self.structural_validation_time_s += time.perf_counter()-started
        self.termination = ("budget_exhausted" if self.oracle.evaluations >= self.oracle.budget else
                            "request_limit" if self.oracle.requests >= self.settings["max_requests"] else
                            "no_legal_action" if not self.actions else None)
        self.done = self.termination is not None
        self.observation = observe(self.workload,self.hardware,self.current_mapping,self.current_record,self.actions,
            current=self.current_objective,best=self.best_objective,
            remaining=self.oracle.budget-self.oracle.evaluations,budget=self.oracle.budget,
            remaining_requests=(self.settings["max_requests"]-self.oracle.requests)/self.settings["max_requests"],
            cached_mapping_hashes=self.oracle.cache,best_mapping_hash=digest(self.best_mapping) if self.best_mapping else '',
            last_reward=self.last_reward)
        self.observation = replace(self.observation,task_features=self.scorer.features(self.current_record,self.feasible(self.best_record)))

    def bottleneck(self):
        if self.current_record is None: return 0.0
        # Critical communication/memory service fraction, derived from the augmented trace DAG.
        diag = self.current_record.diagnostics["criticality"]
        critical = [e for e in self.current_record.schedule_trace if
                    e["job"]["kind"] != "compute" and diag[e["job"]["id"]]["critical"]]
        # Multiple chains are retained; max single critical service is a bounded proxy.
        return max((e["finish_s"]-e["start_s"] for e in critical),default=0)/self.current_record.latency_s

    def phi(self):
        return potential(1.0,self.best_objective,self.bottleneck(),self.settings["lambda_b"],
                         self.settings["lambda_c"],self.done)

    def step(self,action):
        if self.done:
            raise RuntimeError("episode has terminated")
        if action not in self.actions:
            raise ValueError("action excluded by structural mask")
        old_best,old_phi = self.best_objective,self.phi()
        candidate = apply(self.workload,self.hardware,self.current_mapping,action)
        record = self.oracle.evaluate(candidate)
        feasible = self.feasible(record)
        if record.feasible and all(getattr(record,k) <= v for k,v in self.settings["limits"].items()):
            # Physically valid states can exceed vendor limits while exploring.
            self.current_mapping,self.current_record = candidate,record
            self.current_objective = self.scorer.objective(record)
            if self.scorer.better(record,self.best_record):
                self.best_mapping,self.best_record = candidate,record
                self.best_objective = self.current_objective
        self._refresh()
        value = reward(1.0,old_best,self.best_objective,terminal=self.done,
            variant=self.settings["reward_variant"],old_phi=old_phi,new_phi=self.phi())
        self.last_reward = value
        self.observation = replace(self.observation,global_features=(*self.observation.global_features[:-1],value))
        self.trace.append({"action":action,"candidate_hash":record.mapping_hash,"feasible":feasible,
            "current_objective":self.current_objective,"best_objective":self.best_objective,
            "reward":value,"evaluations":self.oracle.evaluations,"requests":self.oracle.requests,
            "cache_hits":self.oracle.cache_hits,"termination":self.termination})
        return self.observation,value,self.done
