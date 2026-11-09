"""Analytical necessary bounds and previously measured feasibility witnesses.

No oracle call, candidate ranking or action mask is performed here.
"""
import gzip
import hashlib
import json
from math import prod
from pathlib import Path
from .mesh import mesh_hardware
from .requirements import PPARequirements
from .serialization import code_hashes, digest


def constraint_check(workload, requirements, counts=tuple(range(1,17))):
    """Necessary conditions only, specific to the current spill-all mesh model."""
    hardware=mesh_hardware(max(counts))
    sizes={key:prod(shape)*workload.tensor_map[op.output].bytes_per_element
           for op in workload.operators for key,shape in workload.parameter_shapes(op).items()}
    read_bound=sum(sizes.values())/sum(p.bandwidth_Bps for p in hardware.memory_ports)
    min_area=min(sum(n.width_mm*n.height_mm for n in mesh_hardware(c).nodes) for c in counts)
    min_static=min(sum(n.static_power_w for n in mesh_hardware(c).nodes)+
                   sum(p.static_power_w for p in mesh_hardware(c).memory_ports) for c in counts)
    warnings=[]
    if requirements.max_latency_s is not None and requirements.max_latency_s < read_bound:
        warnings.append(f'延遲上限 {requirements.max_latency_s*1000:.6g} ms 低於必要參數讀取下界 '
                        f'{read_bound*1000:.6g} ms；目前 spill-all 環境中無可行解。')
    if requirements.max_area_mm2 is not None and requirements.max_area_mm2 < min_area:
        warnings.append(f'面積上限低於最少顆數的 {min_area:g} mm²；目前搜尋空間中無可行解。')
    if requirements.max_power_w is not None and requirements.max_power_w < min_static:
        warnings.append(f'功耗上限低於最少顆數與外部服務的靜態功耗下界 {min_static:g} W；目前環境中無可行解。')
    return {'status':'PROVEN_IMPOSSIBLE' if warnings else 'NOT_EXCLUDED_BY_BOUNDS',
            'warnings':warnings,'parameter_read_lower_bound_s':read_bound,'minimum_area_mm2':min_area,
            'minimum_static_power_w':min_static,'oracle_evaluations':0,'scope':'Necessary bounds only; passing does not prove feasibility. '
            'All constants read at least once; two external ports; no compute/activation/route time counted.'}


def load_tasks(path, models):
    payload=json.loads(Path(path).read_text(encoding='utf-8'))
    if payload['schema']!=1 or payload['input_size']!=64: raise ValueError('unsupported research tasks')
    current=code_hashes()
    if any(current.get(k)!=v for k,v in payload['ppa_source_hashes'].items()):
        raise ValueError('PPA implementation changed; witnesses must be independently revalidated')
    resolved={}
    root=Path(__file__).resolve().parents[2]
    for model in models:
        entry=payload['models'][model]
        data=(root/entry['witness_file']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=entry['witness_sha256']: raise ValueError('witness checksum mismatch')
        witness=json.loads(gzip.decompress(data))
        req=PPARequirements(**entry['requirements']).require_vendor_inputs()
        values=[witness['metrics'][k] for k in ('average_power_w','latency_s','area_mm2')]
        if not witness['physical_feasible'] or any(v>cap for v,cap in zip(values,req.limits)):
            raise ValueError('declared research task has no compliant witness')
        resolved[model]={'requirements':entry['requirements'],'witness':witness,
                         'witness_sha256':entry['witness_sha256']}
    return resolved


def verify_task(workload, task, counts=tuple(range(1,17))):
    req=PPARequirements(**task['requirements'])
    result=constraint_check(workload,req,counts)
    witness=task['witness']
    from .serialization import plain
    structure={k:v for k,v in plain(workload).items() if k!='source_metadata'}
    if digest(structure)!=witness['workload_structure_hash']:
        raise ValueError('workload differs from the measured feasibility witness')
    if len(witness['hardware']['nodes']) not in counts or result['status']=='PROVEN_IMPOSSIBLE':
        raise ValueError('witness contradicts the search space or necessary bounds')
    return {**result,'status':'MEASURED_FEASIBLE_WITNESS',
            'witness_metrics':witness['metrics'],'witness_sha256':task['witness_sha256'],
            'witness_used_to_initialize_agent':False,'calibration_status':'UNCALIBRATED'}
