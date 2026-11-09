"""Layer-by-layer independent reference checks with hashes and declared tolerances."""
from collections import Counter
import time
import numpy as np
import torch
from torch.fx import Interpreter
from .reference import execute
from .tiles import partition,macs,operations
from .serialization import digest,code_hashes
from .frontend import array_hash


class CaptureInterpreter(Interpreter):
    def __init__(self,graph):
        super().__init__(graph)
        self.values = {}

    def run_node(self,node):
        value = super().run_node(node)
        if isinstance(value,torch.Tensor):
            # Capture before later in-place activation can overwrite this tensor.
            self.values[node.name] = value.detach().numpy().copy()
        return value


def validate_model(imported,example,*,tile_count=2,atol=None,rtol=None):
    started = time.perf_counter()
    if type(tile_count) is not int or tile_count < 1:
        raise ValueError("positive reference tile count required")
    dtype = np.float32 if example.dtype == torch.float32 else np.float64
    atol = (1e-4 if dtype == np.float32 else 1e-9) if atol is None else atol
    rtol = (1e-3 if dtype == np.float32 else 1e-7) if rtol is None else rtol
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("finite nonnegative numerical tolerance required")
    w = imported.workload
    counts = {}
    for op in w.operators:
        shape = w.tensor_map[op.output].shape
        counts[op.id] = min(tile_count,shape[2] if len(shape) == 4 else shape[1])
    tiles = partition(w,counts)
    interpreter = CaptureInterpreter(imported.graph_module)
    with torch.no_grad():
        interpreter.run(example)
    values = execute(w,tiles,{imported.input_id:example.numpy()},imported.parameters,dtype=dtype)
    rows = []
    for op in w.operators:
        actual,expected = values[op.output],interpreter.values[op.output]
        error = np.abs(actual-expected)
        finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        mismatches = int(np.count_nonzero(~np.isclose(actual,expected,atol=atol,rtol=rtol)))
        rows.append({"operator":op.id,"kind":op.kind,"shape":list(actual.shape),
            "max_absolute_error":float(error.max()),"mean_absolute_error":float(error.mean()),
            "mismatched_elements":mismatches,"finite":finite,"passed":finite and mismatches == 0,
            "reference_sha256":array_hash(expected),"numpy_sha256":array_hash(actual)})
    return {"status":"PASSED" if all(r['passed'] for r in rows) else "FAILED",
        "scope":"numerical inference/parser verification; no trained accuracy or calibrated PPA claim",
        "model":w.name,"workload_hash":digest(w),"input_sha256":array_hash(example.numpy()),
        "input_value_float32_sha256":array_hash(example.numpy().astype(np.float32)),
        "dtype":str(example.dtype),"atol":atol,"rtol":rtol,"tile_count_requested":tile_count,
        "compiled_tile_count":len(tiles),"operator_count":len(w.operators),
        "operator_kinds":dict(Counter(o.kind for o in w.operators)),
        "parameter_bytes":sum(a.nbytes for a in imported.parameters.values()),
        "total_macs":sum(macs(w,t) for t in tiles),"modeled_operations":sum(operations(w,t) for t in tiles),
        "layers":rows,"source_hashes":code_hashes(),"wall_time_s":time.perf_counter()-started}
