"""Vendor PPA task: hard limits, five preferences and dimensionless scoring."""
from dataclasses import dataclass
import math

PRESET_WEIGHTS = {
    'balanced': (1/3, 1/3, 1/3),
    'power': (0.6, 0.2, 0.2),
    'performance': (0.2, 0.6, 0.2),
    'area': (0.2, 0.2, 0.6),
}
PREFERENCE_LABELS = {'balanced':'PPA 平均', 'power':'偏重功耗',
    'performance':'偏重效能', 'area':'偏重面積', 'custom':'使用者自訂'}
METRICS = ('average_power_w', 'latency_s', 'area_mm2')
TASK_FEATURE_SIZE = 16


@dataclass(frozen=True)
class PPARequirements:
    max_power_w: float | None = None
    max_latency_s: float | None = None
    max_area_mm2: float | None = None
    preference: str = 'balanced'
    weights: tuple | None = None

    def __post_init__(self):
        for value in self.limits:
            if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError('PPA limits must be positive finite numbers')
        if self.preference not in PREFERENCE_LABELS:
            raise ValueError('choose balanced, power, performance, area or custom')
        if self.preference != 'custom' and self.weights is not None:
            raise ValueError('weights can only be supplied with custom preference')
        values = self.weights if self.preference == 'custom' else PRESET_WEIGHTS[self.preference]
        if (not isinstance(values,(tuple,list)) or len(values) != 3 or any(isinstance(v,bool) or not isinstance(v,(int,float))
                or not math.isfinite(v) or v < 0 for v in values) or not 0 < sum(values) < math.inf):
            raise ValueError('custom weights require three finite nonnegative values with positive sum')
        if self.preference == 'custom':
            object.__setattr__(self,'weights',tuple(v/sum(values) for v in values))

    @property
    def limits(self): return (self.max_power_w,self.max_latency_s,self.max_area_mm2)
    @property
    def normalized_weights(self):
        return self.weights if self.preference == 'custom' else PRESET_WEIGHTS[self.preference]

    def require_vendor_inputs(self):
        if any(v is None for v in self.limits):
            raise ValueError('vendor search requires max_power_w, max_latency_s and max_area_mm2')
        return self

    def bind(self,initial=None):
        if initial is None:
            # Declared SI scales, never fitted by testing an unchosen design.
            return PPAScorer(self,tuple(v if v is not None else 1.0 for v in self.limits))
        if not initial.feasible: raise ValueError('physically feasible initial design required')
        scales = tuple(limit if limit is not None else getattr(initial,key)
            for key,limit in zip(METRICS,self.limits))
        return PPAScorer(self,scales)


@dataclass(frozen=True)
class PPAScorer:
    requirements: PPARequirements
    scales: tuple

    def __post_init__(self):
        if len(self.scales) != 3 or any(not math.isfinite(v) or v <= 0 for v in self.scales):
            raise ValueError('positive finite PPA normalization scales required')

    def assess(self,record):
        if record is None:
            return {'feasible':False,'physical_feasible':False,'weighted_cost':None,
                'search_merit':None,'violations':['No physically valid action result has been measured'],
                'metrics':dict.fromkeys(METRICS)}
        metrics = {key:getattr(record,key) for key in METRICS}
        if not record.feasible:
            return {'feasible':False,'physical_feasible':False,'weighted_cost':None,
                'search_merit':None,'violations':list(record.violations),'metrics':metrics}
        values = tuple(metrics.values())
        if any(v is None or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('physically feasible evaluation must contain positive finite PPA metrics')
        cost = math.fsum(w*(v/s) for w,v,s in zip(self.requirements.normalized_weights,values,self.scales))
        excess = tuple(max(0.0,v/limit-1) if limit is not None else 0.0
            for v,limit in zip(values,self.requirements.limits))
        violations = [f'{key}={v:.12g} exceeds limit {limit:.12g}'
            for key,v,limit,e in zip(METRICS,values,self.requirements.limits,excess) if e > 0]
        # Every compliant design scores below 1; every violating design above 2.
        # Within compliant designs merit is strictly monotone in weighted cost.
        merit = cost/(1+cost) if not violations else 2+math.fsum(excess)+cost/(1+cost)
        return {'feasible':not violations,'physical_feasible':True,'weighted_cost':cost,
            'search_merit':merit,'violation_excess':excess,'violations':violations,'metrics':metrics}

    def better(self,candidate,incumbent):
        a,b = self.assess(candidate),self.assess(incumbent)
        if not a['physical_feasible']: return False
        if not b['physical_feasible']: return True
        if a['search_merit'] != b['search_merit']:
            return a['search_merit'] < b['search_merit']
        # Zero custom weights may tie dominated designs. Keep a dominating
        # candidate without changing the vendor's primary weighted objective.
        x,y = tuple(a['metrics'].values()),tuple(b['metrics'].values())
        return all(v <= w for v,w in zip(x,y)) and any(v < w for v,w in zip(x,y))

    def objective(self,record):
        merit = self.assess(record)['search_merit']
        # Unmeasured/invalid has no verified score. 1 is an archive sentinel,
        # not predicted PPA; every physically valid merit maps strictly below it.
        return 1.0 if merit is None else merit/(1+merit)

    def features(self,record,best_feasible):
        values = tuple(getattr(record,k) for k in METRICS) if record is not None else None
        limits = self.requirements.limits
        return (*self.requirements.normalized_weights,
            *(float(v is not None) for v in limits),
            *(math.log(v)/30 for v in self.scales),
            *((math.log(v/s)/30 for v,s in zip(values,self.scales)) if values is not None else (0.0,)*3),
            *((math.log(v/limit)/30 if limit is not None else 0.0 for v,limit in zip(values,limits)) if values is not None else (0.0,)*3),
            float(best_feasible))

    def report(self,record):
        from .serialization import plain
        return {'requirements':plain(self.requirements),
            'preference_label':PREFERENCE_LABELS[self.requirements.preference],
            'weight_order':['power','performance_latency','area'],
            'normalized_weights':self.requirements.normalized_weights,'normalization_scales':dict(zip(METRICS,self.scales)),
            'assessment':self.assess(record),'physically_calibrated':False,
            'power_definition':'total accounted energy / single-inference makespan; average power, not peak power',
            'performance_definition':'single-inference latency upper bound in seconds; lower is better',
            'area_definition':'sum of compute die areas; excludes package and external DRAM'}
