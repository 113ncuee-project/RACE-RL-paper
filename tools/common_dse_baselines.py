"""Random and generic SA over complete CNN designs using the common oracle."""
import math
import random
import time
import numpy as np
from chiplet_gym_dse import domain,decode
from gemini_common_dse import propose
from race_rl.accelerators.cnn_design import CNNDesign,LayerPlan
from race_rl.serialization import digest


def run(evaluator, *, method, seed):
    if method not in ('random','local_random','sa'): raise ValueError('choose random, local_random, or sa')
    started=time.perf_counter()
    rng=np.random.default_rng(seed)
    workload=evaluator.environment.workload
    indices,sizes=domain(workload)
    current_score=None
    current = CNNDesign(2,tuple(LayerPlan() for _ in workload.operators))
    local_rng=random.Random(seed)
    while evaluator.requests<evaluator.budget:
        parent_hash=digest(current) if method!='random' else None
        if method=='random':
            design=decode(workload,indices,rng.integers(0,sizes))
        elif evaluator.requests==0:
            design=current
        else:
            design,_=propose(workload,current,local_rng)
        result=evaluator.evaluate(design)
        assessment=result.get('assessment')
        score=assessment.get('search_merit') if assessment else None
        fraction=(evaluator.requests-1)/max(1,evaluator.budget-1)
        temperature=.1*(.01/.1)**fraction
        accepted=score is not None and (method in ('random','local_random') or current_score is None or
                 score<=current_score or local_rng.random()<math.exp((current_score-score)/temperature))
        if accepted:
            current,current_score=design,score
        result.update(accepted=bool(accepted),temperature=temperature if method=='sa' else None)
        result.update(proposal_parent_hash=parent_hash,
                      accepted_design_hash=digest(current) if method!='random' else None)
    return {'method':method,'seed':seed,'wall_time_s':time.perf_counter()-started,
            'result':evaluator.snapshot(),
            'method_limits':(['independently sampled full designs'] if method=='random' else
                ['local proposals from the current accepted design, using the shared restricted mutation operators'])+
                ['generic reference, not paper implementation']}
