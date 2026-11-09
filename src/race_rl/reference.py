"""Independent NumPy tile execution for semantic validation, never PPA/search.

The executor implements mathematical inference semantics without calling Torch
operators. float64 is the default; full-model float32 checks declare tolerances.
"""
from math import ceil
import numpy as np
from .tiles import validate_coverage


def execute(workload,tiles,inputs,weights,dtype=np.float64):
    workload.validate()
    validate_coverage(workload,tiles)
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32),np.dtype(np.float64)):
        raise ValueError("reference executor supports floating-point float32/float64 only")
    tensors = {name:np.asarray(data,dtype=dtype) for name,data in inputs.items()}
    for name in set(workload.tensor_map)-set(workload.producers):
        if name not in tensors or tensors[name].shape != workload.tensor_map[name].shape:
            raise ValueError("missing/invalid numerical input")

    def parameter(name,shape):
        raw = weights.get(name)
        if raw is None and name.startswith("weights:"):
            raw = weights.get(name.split(":",1)[1])  # backward-compatible toy weight dictionary
        if raw is None:
            raise ValueError(f"missing numerical parameter {name}")
        array = np.asarray(raw,dtype=dtype)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"invalid numerical parameter {name}")
        return array

    for op in workload.operators:
        shape = workload.tensor_map[op.output].shape
        out = np.zeros(shape,dtype=dtype)
        ins = [tensors[t] for t in op.inputs]
        params = {name:parameter(name,s) for name,s in workload.parameter_shapes(op).items()}
        if op.kind == "BatchNorm" and (params[f"variance:{op.id}"] < 0).any():
            raise ValueError("negative BatchNorm running variance")
        if op.kind == "Conv":
            x = np.pad(ins[0],((0,0),(0,0),(op.padding[0],op.padding[0]),(op.padding[1],op.padding[1])))
            effective = tuple(d*(k-1)+1 for d,k in zip(op.dilation,op.kernel))
            windows = np.lib.stride_tricks.sliding_window_view(x,effective,axis=(2,3))
            windows = windows[:,:,::op.stride[0],::op.stride[1],::op.dilation[0],::op.dilation[1]]
        for tile in (t for t in tiles if t.operator == op.id):
            slices = tuple(slice(a,b) for a,b in tile.output.bounds)
            n,k = slices[:2]
            if op.kind == "Add":
                value = ins[0][slices]+ins[1][slices]
            elif op.kind == "Linear":
                value = ins[0][n] @ params[f"weights:{op.id}"][k].T
                if op.bias: value = value+params[f"bias:{op.id}"][k]
            elif op.kind in {"ReLU","ReLU6"}:
                value = np.maximum(ins[0][slices],0)
                if op.kind == "ReLU6": value = np.minimum(value,6)
            elif op.kind == "Identity":
                value = ins[0][slices]
            elif op.kind == "Flatten":
                value = ins[0].reshape(ins[0].shape[0],-1)[slices]
            elif op.kind == "BatchNorm":
                channel_shape = (1,-1) if len(shape) == 2 else (1,-1,1,1)
                mean = params[f"mean:{op.id}"][k].reshape(channel_shape)
                var = params[f"variance:{op.id}"][k].reshape(channel_shape)
                value = (ins[0][slices]-mean)/np.sqrt(var+op.epsilon)
                if op.affine:
                    value = value*params[f"scale:{op.id}"][k].reshape(channel_shape)+params[f"shift:{op.id}"][k].reshape(channel_shape)
            elif op.kind == "Conv":
                y,x = slices[2:]
                value = np.empty(tuple(b-a for a,b in tile.output.bounds),dtype=dtype)
                channels_in,channels_out = ins[0].shape[1]//op.groups,shape[1]//op.groups
                ka,kb = tile.output.bounds[1]
                for group in range(op.groups):
                    a,b = max(ka,group*channels_out),min(kb,(group+1)*channels_out)
                    if a >= b: continue
                    data = windows[n,group*channels_in:(group+1)*channels_in,y,x]
                    value[:,a-ka:b-ka] = np.einsum('ncyxij,kcij->nkyx',data,
                        params[f"weights:{op.id}"][a:b],optimize=True)
                if op.bias: value = value+params[f"bias:{op.id}"][k].reshape(1,-1,1,1)
            else:
                value = np.empty(tuple(b-a for a,b in tile.output.bounds),dtype=dtype)
                ha,hb = tile.output.bounds[2]
                wa,wb = tile.output.bounds[3]
                for y in range(ha,hb):
                    for x in range(wa,wb):
                        if op.kind == "AdaptiveAvgPool":
                            a,b = y*ins[0].shape[2]//shape[2],ceil((y+1)*ins[0].shape[2]/shape[2])
                            c,d = x*ins[0].shape[3]//shape[3],ceil((x+1)*ins[0].shape[3]/shape[3])
                            cell = ins[0][n,k,a:b,c:d].mean(axis=(2,3))
                        else:
                            sy,sx = y*op.stride[0]-op.padding[0],x*op.stride[1]-op.padding[1]
                            iy = [sy+j*op.dilation[0] for j in range(op.kernel[0]) if 0 <= sy+j*op.dilation[0] < ins[0].shape[2]]
                            ix = [sx+j*op.dilation[1] for j in range(op.kernel[1]) if 0 <= sx+j*op.dilation[1] < ins[0].shape[3]]
                            if not iy or not ix: raise ValueError("all-padding pooling window")
                            patch = ins[0][n,k][:,:,iy][:,:,:,ix]
                            if op.kind == "MaxPool":
                                cell = patch.max(axis=(2,3))
                            else:
                                if op.divisor_override is not None:
                                    divisor = op.divisor_override
                                elif op.count_include_pad:
                                    divisor = (min(sy+op.kernel[0],ins[0].shape[2]+op.padding[0])-max(sy,-op.padding[0]))*(min(sx+op.kernel[1],ins[0].shape[3]+op.padding[1])-max(sx,-op.padding[1]))
                                else:
                                    divisor = len(iy)*len(ix)
                                cell = patch.sum(axis=(2,3))/divisor
                        value[:,:,y-ha,x-wa] = cell
            out[slices] = value
        tensors[op.output] = out
    return tensors
