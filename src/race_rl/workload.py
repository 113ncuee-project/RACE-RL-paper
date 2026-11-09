"""Explicit floating-point inference DAG; spatial tensors use NCHW, Linear uses NF."""
from dataclasses import dataclass
from math import prod, isfinite
from functools import cached_property
from types import MappingProxyType

OPERATOR_KINDS = ("Conv", "Add", "MaxPool", "AvgPool", "Linear", "ReLU", "ReLU6",
                  "BatchNorm", "Flatten", "AdaptiveAvgPool", "Identity")


@dataclass(frozen=True)
class Tensor:
    id: str
    shape: tuple[int, ...]
    bytes_per_element: int = 4

    def __post_init__(self):
        object.__setattr__(self,"shape",tuple(self.shape))
        if not self.id or not self.shape or any(type(x) is not int or x <= 0 for x in self.shape):
            raise ValueError("tensor shape must contain positive integers")
        if type(self.bytes_per_element) is not int or self.bytes_per_element <= 0:
            raise ValueError("invalid precision")

    @property
    def bytes(self):
        return prod(self.shape) * self.bytes_per_element


@dataclass(frozen=True)
class Operator:
    id: str
    kind: str
    inputs: tuple[str, ...]
    output: str
    kernel: tuple[int, int] = (1, 1)
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)
    dilation: tuple[int, int] = (1, 1)
    groups: int = 1
    bias: bool = False
    epsilon: float = 1e-5
    affine: bool = True
    ceil_mode: bool = False
    count_include_pad: bool = True
    divisor_override: int | None = None

    def __post_init__(self):
        for name in ("inputs","kernel","stride","padding","dilation"):
            object.__setattr__(self,name,tuple(getattr(self,name)))


def pool_extent(size, kernel, stride, padding, dilation=1, ceil_mode=False):
    numerator = size + 2*padding - dilation*(kernel-1)-1
    extent = (numerator + (stride-1 if ceil_mode else 0))//stride+1
    if ceil_mode and (extent-1)*stride >= size+padding:
        extent -= 1
    return extent


@dataclass(frozen=True)
class Workload:
    tensors: tuple[Tensor, ...]
    operators: tuple[Operator, ...]
    outputs: tuple[str, ...]
    name: str = "custom"
    source_metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        for name in ("tensors","operators","outputs"):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        metadata = tuple(tuple(pair) for pair in self.source_metadata)
        if any(len(pair) != 2 or not all(isinstance(v,str) for v in pair) for pair in metadata):
            raise ValueError("source metadata must contain immutable string pairs")
        object.__setattr__(self,"source_metadata",metadata)

    @cached_property
    def tensor_map(self):
        return MappingProxyType({t.id: t for t in self.tensors})

    @cached_property
    def operator_map(self):
        return MappingProxyType({o.id: o for o in self.operators})

    @cached_property
    def producers(self):
        return MappingProxyType({o.output: o.id for o in self.operators})

    def validate(self):
        ts = self.tensor_map
        if any(len(t.shape) not in (2,4) for t in self.tensors):
            raise ValueError("workload supports NF/NCHW tensor ranks only")
        if len(ts) != len(self.tensors) or len(self.operator_map) != len(self.operators):
            raise ValueError("duplicate IDs")
        if not self.operators or not self.outputs or not set(self.outputs) <= set(ts):
            raise ValueError("missing operators or outputs")
        if len(self.producers) != len(self.operators):
            raise ValueError("multiple output owners")
        available = set(ts) - set(self.producers)
        for o in self.operators:
            if not o.id or not o.inputs or not set(o.inputs) <= available or o.output not in ts:
                raise ValueError("operators must be topologically ordered with known tensor dependencies")
            if o.kind not in OPERATOR_KINDS:
                raise ValueError(f"unsupported operator: {o.kind}")
            ins = [ts[t].shape for t in o.inputs]
            out = ts[o.output].shape
            if o.kind == "Add":
                if len(ins) != 2 or ins[0] != ins[1] or out != ins[0]:
                    raise ValueError("Add requires two equal shapes; broadcasting unsupported")
            else:
                if len(ins) != 1:
                    raise ValueError(f"{o.kind} expects exactly one input")
                if o.kind in {"ReLU", "ReLU6", "Identity", "BatchNorm"}:
                    if out != ins[0]:
                        raise ValueError("shape-preserving operator changed shape")
                    if o.kind == "BatchNorm" and (len(out) not in (2,4) or not isfinite(o.epsilon) or o.epsilon <= 0):
                        raise ValueError("BatchNorm expects NF/NCHW and positive finite epsilon")
                elif o.kind == "Flatten":
                    if len(ins[0]) not in (2,4) or out != (ins[0][0], prod(ins[0][1:])):
                        raise ValueError("Flatten supports only complete nonbatch dimensions")
                elif o.kind == "Linear":
                    if len(ins[0]) != 2 or len(out) != 2 or out[0] != ins[0][0]:
                        raise ValueError("Linear expects NF; flatten must be explicit")
                else:
                    if len(ins[0]) != 4 or len(out) != 4 or out[0] != ins[0][0]:
                        raise ValueError("spatial operator expects NCHW")
                    if o.kind == "AdaptiveAvgPool":
                        if out[1] != ins[0][1]:
                            raise ValueError("adaptive pooling must preserve channels")
                    else:
                        if any(len(v) != 2 for v in (o.kernel,o.stride,o.padding,o.dilation)):
                            raise ValueError("spatial attributes must be integer pairs")
                        if any(type(x) is not int or x <= 0 for x in (*o.kernel,*o.stride,*o.dilation)) or any(
                                type(x) is not int or x < 0 for x in o.padding):
                            raise ValueError("invalid spatial attributes")
                        if o.kind == "Conv" and o.ceil_mode:
                            raise ValueError("Conv cannot use ceil_mode")
                        expected = tuple(pool_extent(n,k,s,p,d,o.ceil_mode) for n,k,s,p,d in
                                         zip(ins[0][2:],o.kernel,o.stride,o.padding,o.dilation))
                        if out[2:] != expected:
                            raise ValueError("invalid spatial output shape")
                        if o.kind == "Conv":
                            if type(o.groups) is not int or o.groups < 1 or ins[0][1] % o.groups or out[1] % o.groups:
                                raise ValueError("group count must divide input/output channels")
                        else:
                            if out[1] != ins[0][1] or any(p > k//2 for p,k in zip(o.padding,o.kernel)):
                                raise ValueError("invalid pooling channels/padding")
                            if o.kind == "AvgPool" and o.dilation != (1,1):
                                raise ValueError("AvgPool dilation unsupported")
                            if o.divisor_override is not None and (type(o.divisor_override) is not int or o.divisor_override <= 0):
                                raise ValueError("invalid average pooling divisor")
            if len({ts[t].bytes_per_element for t in (*o.inputs,o.output)}) != 1:
                raise ValueError("mixed precision unsupported")
            available.add(o.output)
        return True

    def weight_shape(self, op):
        ts = self.tensor_map
        if op.kind == "Conv":
            return (ts[op.output].shape[1],ts[op.inputs[0]].shape[1]//op.groups,*op.kernel)
        if op.kind == "Linear":
            return (ts[op.output].shape[1],ts[op.inputs[0]].shape[1])
        return ()

    def parameter_shapes(self, op):
        """All read-only inference constants; no bias/BN traffic is silently omitted."""
        shapes = {}
        weight = self.weight_shape(op)
        channels = self.tensor_map[op.output].shape[1]
        if weight:
            shapes[f"weights:{op.id}"] = weight
            if op.bias:
                shapes[f"bias:{op.id}"] = (channels,)
        if op.kind == "BatchNorm":
            shapes.update({f"mean:{op.id}":(channels,),f"variance:{op.id}":(channels,)})
            if op.affine:
                shapes.update({f"scale:{op.id}":(channels,),f"shift:{op.id}":(channels,)})
        return shapes

    @cached_property
    def parameter_map(self):
        return MappingProxyType({name:shape for op in self.operators for name,shape in self.parameter_shapes(op).items()})
