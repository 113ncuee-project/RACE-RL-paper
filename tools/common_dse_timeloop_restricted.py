"""Restricted hybrid-style proposal, not native Timeloop mapper."""
from dataclasses import replace
import random
import time
from race_rl.accelerators.cnn_design import CNNDesign,LayerPlan,PROFILES


def run(evaluator, *, seed, block_size=8):
    rng=random.Random(seed);started=time.perf_counter();ops=evaluator.environment.workload.operators
    while evaluator.requests<evaluator.budget:
        # Profile vector plays the role of a factorization; the common CNNDesign
        # has no loop-order permutation, so enumerate supported placement/residency.
        count=rng.randint(1,16); profiles=[rng.choice(PROFILES) for _ in ops]
        for option in range(block_size):
            if evaluator.requests>=evaluator.budget:break
            plans=tuple(LayerPlan(*profile,(i+option//4)%count,bool((option//2)%2),
                                  bool(option%2),bool((option//4)%2))
                        for i,profile in enumerate(profiles))
            result=evaluator.evaluate(CNNDesign(count,plans))
            result.update(phase='restricted_profile_group',group_option=option)
    return {'method':'timeloop_hybrid_restricted','seed':seed,'wall_time_s':time.perf_counter()-started,
        'result':evaluator.snapshot(),'method_limits':['Timeloop hybrid-inspired grouping, not native Timeloop',
        'CNNDesign lacks native loop permutations/index factorization and Timeloop mapspace',
        'uniform random profiles and 8 supported placement/residency variations, all charged']}
