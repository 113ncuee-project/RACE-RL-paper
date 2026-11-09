"""Objective plateau detection; finite-search progress is not a convergence proof."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PlateauSettings:
    min_episodes: int = 20
    patience: int = 15
    relative_improvement: float = .001

    def __post_init__(self):
        if type(self.min_episodes) is not int or self.min_episodes<1 or type(self.patience) is not int or self.patience<1:
            raise ValueError('positive warmup and patience required')
        if not math.isfinite(self.relative_improvement) or not 0<self.relative_improvement<1:
            raise ValueError('relative improvement must be between zero and one')


class ObjectivePlateau:
    def __init__(self,settings):
        self.settings=settings;self.reference=None;self.compliant=None;self.stale=0

    def update(self,assessment,episode):
        feasible=assessment['feasible']
        value=assessment['weighted_cost'] if feasible else assessment['search_merit']
        if value is None:
            self.stale+=1
            return episode>=self.settings.min_episodes and self.stale>=self.settings.patience
        # Feasibility acquisition is always a material improvement. Small
        # improvements accumulate against the last significant reference.
        if (self.reference is None or feasible!=self.compliant or
            (self.reference-value)/self.reference >= self.settings.relative_improvement):
            self.reference=value;self.compliant=feasible;self.stale=0
        else:self.stale+=1
        return episode>=self.settings.min_episodes and self.stale>=self.settings.patience
