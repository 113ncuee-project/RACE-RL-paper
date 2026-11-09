"""Discrete Gaussian-process Bayesian optimization for shared CNN designs."""
import math
import random
import time
import numpy as np
from gemini_common_dse import propose
from race_rl.accelerators.cnn_design import CNNDesign, LayerPlan, PROFILES


def features(design):
    result=[design.compute_count/16.0]
    for p in design.plans:
        result.extend((PROFILES.index((p.h,p.w,p.c,p.k))/3.0,p.target/15.0,
                       float(p.spread),float(p.keep_output),float(p.keep_weights)))
    return np.asarray(result,dtype=float)


def acquisition(train_x, train_y, pool):
    """Fixed-kernel GP and analytic expected improvement for minimization."""
    x=np.asarray(train_x);y=np.asarray(train_y);z=np.asarray(pool)
    def kernel(a,b):
        dist=((a[:,None,:]-b[None,:,:])**2).mean(axis=2)
        return np.exp(-dist/.12)
    centre=float(y.mean());scale=max(float(y.std()),1e-6);v=(y-centre)/scale
    k=kernel(x,x)+np.eye(len(x))*1e-6
    chol=np.linalg.cholesky(k);alpha=np.linalg.solve(chol.T,np.linalg.solve(chol,v))
    cross=kernel(x,z);mu=cross.T@alpha
    proj=np.linalg.solve(chol,cross);sigma=np.sqrt(np.maximum(1.0-(proj*proj).sum(axis=0),1e-12))
    improvement=float(v.min())-mu;u=improvement/sigma
    cdf=np.array([.5*(1+math.erf(float(a)/math.sqrt(2))) for a in u])
    return improvement*cdf+sigma*np.exp(-u*u/2)/math.sqrt(2*math.pi)


def run(evaluator, *, seed, initial_points=4, pool_size=32):
    start=time.perf_counter();rng=random.Random(seed);workload=evaluator.environment.workload
    initial=CNNDesign(2,tuple(LayerPlan() for _ in workload.operators));observed=[]
    while evaluator.requests<evaluator.budget:
        if evaluator.requests==0: candidate=initial;phase='initial'
        elif len(observed)<initial_points:
            candidate,_=propose(workload,initial,rng);phase='initial_mutation'
        else:
            # Generate candidates without PPA evaluations; all oracle calls are charged.
            ranked=sorted(observed,key=lambda row:row[1]);pool=[]
            for i in range(pool_size):
                parent=rng.choice(ranked[:min(8,len(ranked))])[0] if i<pool_size*3//4 else initial
                candidate,_=propose(workload,parent,rng);pool.append(candidate)
            scores=acquisition([features(x) for x,_ in observed],[s for _,s in observed],
                               [features(x) for x in pool])
            candidate=pool[int(np.argmax(scores))];phase='gp_expected_improvement'
        item=evaluator.evaluate(candidate);assessment=item.get('assessment') or {}
        merit=assessment.get('search_merit');score=float(merit) if merit is not None else 10.0
        item.update(phase=phase, surrogate_score=score);observed.append((candidate,score))
    return {'method':'bayesopt_gp_ei','seed':seed,'wall_time_s':time.perf_counter()-start,
      'result':evaluator.snapshot(),'method_limits':['generic fixed-kernel discrete GP/EI, not a reproduction of a named BO paper',
      'local candidate pool 32; invalid physical proposals assigned surrogate score 10',
      'all evaluated initial, invalid and cached proposals count toward budget']}
