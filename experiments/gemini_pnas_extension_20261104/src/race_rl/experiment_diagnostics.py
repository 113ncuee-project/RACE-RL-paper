"""Small experiment metadata and non-learning comparison controllers."""
import importlib.metadata
import math
import os
import platform
import random
import subprocess
import sys
import time


def platform_snapshot():
    def query(command):
        try: return subprocess.check_output(command,text=True,stderr=subprocess.DEVNULL,timeout=5).strip()
        except (OSError,subprocess.SubprocessError): return None
    packages={}
    for name in ('race-rl','torch','torchvision','numpy','matplotlib'):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    return {'os':platform.system(),'os_version':platform.version(),'architecture':platform.machine(),
            'python':sys.version,'packages':packages,'logical_cpu_count':os.cpu_count(),
            'mac_model':query(['sysctl','-n','hw.model']) if sys.platform=='darwin' else None,
            'cpu':query(['sysctl','-n','machdep.cpu.brand_string']) if sys.platform=='darwin' else platform.processor(),
            'memory_bytes':query(['sysctl','-n','hw.memsize']) if sys.platform=='darwin' else None,
            'git_head':query(['git','rev-parse','HEAD']),
            'git_worktree_dirty':bool(query(['git','status','--porcelain'])),
            'device':'CPU','torch_threads':1,'precision':'float64','deterministic_algorithms':True,
            'not_measured':['CPU utilization','peak RSS','thermal throttling','sleep/power state']}


def comparison_episode(env, method, seed):
    """Uniform legal-action random walk or SA; every proposal is charged.

    Same complete coupled action support, reset and request ceiling as RL.
    SA may reject the already measured current candidate; it retains the best
    measured archive. There is no uncharged neighbor lookahead.
    """
    if method not in ('random','sa'): raise ValueError('unsupported comparison')
    rng=random.Random(seed)
    env.reset()
    started=time.perf_counter()
    while not env.done:
        old=(env.hardware,env.compute_count,env.current_mapping,env.current_record,env.current_objective)
        action=rng.choice(env.actions)
        env.step(action)
        accepted=True
        if method=='sa' and env.current_objective>old[-1]:
            fraction=(env.oracle.requests-1)/env.settings['max_requests']
            temperature=.2*(.01/.2)**fraction
            accepted=rng.random()<math.exp(-(env.current_objective-old[-1])/temperature)
            if not accepted:
                env.hardware,env.compute_count,env.current_mapping,env.current_record,env.current_objective=old
                env._refresh()
                env.trace[-1].update(compute_count=env.compute_count,current_objective=env.current_objective)
        env.trace[-1]['controller_accepted']=accepted
        if hasattr(env,'progress_path'):
            from .serialization import save_json
            save_json(env.progress_path,{**getattr(env,'progress_context',{}),'stage':'COMPARISON',
                'requests':env.oracle.requests,'evaluations':env.oracle.evaluations,
                'cache_hits':env.oracle.cache_hits,'best_objective':env.best_objective})
    return {'steps':len(env.trace),'oracle_requests':env.oracle.requests,
            'oracle_evaluations':env.oracle.evaluations,'cache_hits':env.oracle.cache_hits,
            'best_objective':env.best_objective,'return':sum(r['reward'] for r in env.trace),
            'termination':env.termination,'parameter_updates':0,'algorithm':method,'critic_used':False,
            'environment_step_time_s':time.perf_counter()-started,
            'sa_temperature':'0.2 -> 0.01 geometric within each episode' if method=='sa' else None}
