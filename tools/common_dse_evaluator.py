"""Action-independent, request-budgeted evaluator for complete CNN designs.

The supported design representation remains CNNDesign; this module does not
claim it represents all native Gemini or Chiplet-Gym architectures.
"""
from dataclasses import asdict
from time import perf_counter

from race_rl.accelerators.cnn_design import CNNDesign,LayerPlan,PROFILES
from race_rl.accelerators.cnn_environment import CNNOracle
from race_rl.oracle import BudgetExhausted
from race_rl.serialization import digest


def encode_design(design):
    return {'version':1,'compute_count':design.compute_count,'plans':[
        [PROFILES.index((p.h,p.w,p.c,p.k)),int(p.keep_output)+2*int(p.keep_weights),
         int(p.spread),p.target] for p in design.plans]}


def decode_design(encoded):
    if encoded.get('version') != 1:
        raise ValueError('unsupported design encoding')
    plans=[]
    for row in encoded['plans']:
        if (len(row)!=4 or any(type(v) is not int for v in row) or
                not 0<=row[0]<len(PROFILES) or not 0<=row[1]<4 or row[2] not in (0,1)):
            raise ValueError('invalid encoded layer plan')
        profile,residency,spread,target=row
        plans.append(LayerPlan(*PROFILES[profile],target,bool(spread),bool(residency&1),bool(residency&2)))
    return CNNDesign(encoded['compute_count'],tuple(plans))


class CommonDSEEvaluator:
    def __init__(self, environment, budget):
        if type(budget) is not int or budget < 1:
            raise ValueError('positive request budget required')
        if environment.cost_profile is None:
            raise ValueError('common DSE requires a valid PPA cost profile')
        environment.requirements.require_vendor_inputs()
        self.environment = environment
        self.scorer = environment.requirements.bind()
        self.oracle = CNNOracle(environment.model, environment.cost_profile, budget)
        self.budget = budget
        self.requests = 0
        self.best_design = self.best_record = None
        self.trace = []

    def evaluate(self, design):
        if self.requests >= self.budget:
            raise BudgetExhausted('common DSE request budget exhausted')
        self.requests += 1
        started = perf_counter()
        if not isinstance(design, CNNDesign) or len(design.plans) != len(self.environment.workload.operators):
            result = {'status':'STRUCTURALLY_INVALID', 'physical_feasible':False, 'assessment':None,
                      'design_hash':None, 'cached':False,
                      'violations':['complete CNNDesign with one plan per operator required']}
        else:
            before = self.oracle.cache_hits
            record = self.oracle.evaluate(design)
            assessment = self.scorer.assess(record)
            if self.scorer.better(record, self.best_record):
                self.best_design, self.best_record = design, record
            result = {'status':record.evaluation_status, 'physical_feasible':record.physical_feasible,
                      'assessment':assessment, 'design_hash':record.design_hash,
                      'candidate_encoding':encode_design(design),
                      'cached':self.oracle.cache_hits > before, 'violations':list(record.violations)}
        result.update(request=self.requests, actual_evaluations=self.oracle.evaluations,
                      cache_hits=self.oracle.cache_hits, elapsed_s=perf_counter()-started)
        self.trace.append(result)
        return result

    def snapshot(self):
        return {'requests':self.requests, 'actual_evaluations':self.oracle.evaluations,
                'cache_hits':self.oracle.cache_hits, 'budget':self.budget,
                'best_assessment':self.scorer.assess(self.best_record),
                'best_design':asdict(self.best_design) if self.best_design else None,
                'best_feedback':self.best_record.model_feedback if self.best_record else None,
                'trace':list(self.trace),
                'environment_identity':digest({'workload':self.environment.workload,
                    'requirements':self.environment.requirements,'memory':self.environment.memory.record(),
                    'platform':self.environment.platform,'cost_profile':self.environment.cost_profile})}
