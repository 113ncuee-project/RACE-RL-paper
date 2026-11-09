"""Controller comparisons on the same action space, oracle and initial mapping."""
import math
import random
from .actions import apply,legal_actions


def run_baseline(env,method,seed=7):
    if method not in {"random","greedy","sa"}:
        raise ValueError("unknown baseline")
    rng = random.Random(seed)
    env.reset()
    steps = 0
    while not env.done:
        if method == "greedy":
            old_current = env.current_objective
            scored = []
            # Evaluate all possible neighbors until the shared budget runs out.
            # Cache lookups are requests and are included in the explicit safety ceiling.
            for a in env.actions:
                if env.oracle.evaluations >= env.oracle.budget or env.oracle.requests >= env.settings["max_requests"]:
                    break
                candidate = apply(env.workload,env.hardware,env.current_mapping,a)
                record = env.oracle.evaluate(candidate)
                if env.feasible(record):
                    cost = env.scorer.objective(record)
                    scored.append((cost,a,candidate,record))
                    if env.scorer.better(record,env.best_record):
                        env.best_objective,env.best_mapping,env.best_record = cost,candidate,record
            if scored:
                cost,a,candidate,record = min(scored,key=lambda row:row[0])
                if cost < env.current_objective:
                    env.current_objective,env.current_mapping,env.current_record = cost,candidate,record
            env._refresh()
            if not env.done and env.current_objective >= old_current:
                env.done,env.termination = True,"local_minimum"
            env.trace.append({"best_objective":env.best_objective,"current_objective":env.current_objective,
                "evaluations":env.oracle.evaluations,"requests":env.oracle.requests,"termination":env.termination})
        else:
            a = rng.choice(env.actions)
            old = (env.current_mapping,env.current_record,env.current_objective)
            _,_,_ = env.step(a)
            if method == "sa" and env.current_objective > old[2]:
                temperature = max(0.01,0.2*0.95**steps)
                if rng.random() >= math.exp(-math.log(env.current_objective/old[2])/temperature):
                    env.current_mapping,env.current_record,env.current_objective = old
                    env._refresh()
                    env.trace[-1]["current_objective"] = env.current_objective
                    env.trace[-1]["accepted"] = False
        steps += 1
    return {"method":method,"seed":seed,"best_objective":env.best_objective,
            "oracle_evaluations":env.oracle.evaluations,"oracle_requests":env.oracle.requests,
            "cache_hits":env.oracle.cache_hits,"invalid_proposals":env.oracle.invalid_proposals,
            "termination":env.termination,"training_evaluations":0}
