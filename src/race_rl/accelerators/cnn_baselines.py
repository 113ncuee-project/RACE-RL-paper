"""Random and simulated-annealing controllers for the CNN transaction space.

These are generic references, not implementations of any chiplet paper.
Every proposal goes through the same environment step as the learned policy.
"""
import math
import random
import time

from .cnn_design import PROFILES, coupled_scope, transaction


def random_transaction(observation, rng):
    count = rng.randrange(1, 17)
    anchor = rng.choice(observation.anchor_indices)
    scope = coupled_scope(observation.workload, anchor, observation.max_scope)
    choices = [count - 1, anchor]
    for _ in scope:
        choices.extend((rng.randrange(len(PROFILES)), rng.randrange(4),
                        rng.randrange(2), rng.randrange(count)))
    return transaction(observation.workload, observation.current_design,
                       tuple(choices), observation.max_scope)


def run_cnn_baseline(env, method, *, seed=7, initial_temperature=0.1,
                     final_temperature=0.01):
    """Run exactly one request-budgeted episode, including invalid/cache hits."""
    if method not in ('random', 'sa'):
        raise ValueError('method must be random or sa')
    if not (math.isfinite(initial_temperature) and math.isfinite(final_temperature)
            and initial_temperature > 0 and final_temperature > 0):
        raise ValueError('temperatures must be positive and finite')
    if env.cost_profile is None:
        raise ValueError('a PPA cost profile is required for a scored comparison')
    rng = random.Random(seed)
    start = time.perf_counter()
    observation = env.reset()
    while not env.done:
        action = random_transaction(observation, rng)
        previous = (env.current_design, env.compute_count, env.current_record,
                    env.current_feedback, env.current_objective)
        observation, _, _ = env.step(action)
        accepted = env.trace[-1]['physical_feasible']
        # Physical failures do not change the current design. The best archive
        # includes every evaluated candidate, even when SA rejects the move.
        if method == 'sa' and env.current_record is not previous[2] and previous[2] is not None:
            delta = env.current_objective - previous[4]
            fraction = (env.oracle.requests - 1) / max(1, env.budget - 1)
            temperature = initial_temperature * (final_temperature / initial_temperature) ** fraction
            accepted = delta <= 0 or rng.random() < math.exp(-delta / temperature)
            if not accepted:
                (env.current_design, env.compute_count, env.current_record,
                 env.current_feedback, env.current_objective) = previous
                env._refresh()
                observation = env.observation
        env.trace[-1]['accepted'] = accepted
        if not accepted:
            env.trace[-1]['state_after'].update(
                design_hash=env.trace[-1]['state_before']['design_hash'],
                compute_count=env.compute_count, objective=env.current_objective,
                ppa_valid=env.current_record is not None)
        env.trace[-1]['accepted_design_hash'] = (
            env.current_record.design_hash if env.current_record else None)
    return {'method': method, 'seed': seed, 'oracle_requests': env.oracle.requests,
            'oracle_evaluations': env.oracle.evaluations, 'cache_hits': env.oracle.cache_hits,
            'wall_time_s': time.perf_counter() - start, 'termination': env.termination,
            'assessment': env.scorer.assess(env.best_record),
            'best_design': env.best_design,
            'best_feedback': env.best_record.model_feedback if env.best_record else None,
            'best_ppa_detail': env.best_record.ppa_detail if env.best_record else None,
            'trace': env.trace}
