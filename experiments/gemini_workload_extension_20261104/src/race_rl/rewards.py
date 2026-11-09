"""Verified-best finite-episode returns; gamma is exactly one throughout."""
import math


def improvement(initial,best):
    if not 0 < best <= initial or not math.isfinite(initial):
        raise ValueError("invalid verified best objective")
    return math.log(initial/best)


def potential(initial,best,bottleneck=0.0,lambda_b=1.0,lambda_c=0.0,terminal=False):
    if terminal:
        return 0.0
    if not math.isfinite(bottleneck) or bottleneck < 0:
        raise ValueError("invalid bottleneck proxy")
    return lambda_b*improvement(initial,best)-lambda_c*bottleneck


def reward(initial,old_best,new_best,*,terminal=False,variant="dense",old_phi=0.0,new_phi=0.0):
    if new_best > old_best:
        raise ValueError("best verified archive must be monotone")
    if variant == "dense":
        return math.log(old_best/new_best)
    base = improvement(initial,new_best) if terminal else 0.0
    if variant == "terminal":
        return base
    if variant == "potential":
        return base+(0.0 if terminal else new_phi)-old_phi
    raise ValueError("unknown reward variant")
