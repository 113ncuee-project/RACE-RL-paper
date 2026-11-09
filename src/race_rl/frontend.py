"""Independent CPU Torch FX importer with strict supported inference semantics.

No Legacy parser, post-op fusion, weights download or ignored tensor operator is
used. Each FX tensor-producing node becomes an explicit operator and tensor.
"""
from dataclasses import dataclass
import hashlib
import operator as python_operator
import numpy as np
import torch
from torch import nn
from torch.fx import Node as FXNode, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import torch.nn.functional as F
from .workload import Workload,Tensor,Operator
from .serialization import digest

SUPPORTED_MODELS = ("resnet18","resnet34","resnet50","vgg16","vgg19","mobilenet_v2","alexnet")


class UnsupportedOperator(ValueError):
    pass


@dataclass
class ImportedModel:
    workload: Workload
    parameters: dict
    parameter_manifest: dict
    graph_module: torch.fx.GraphModule
    input_id: str


def array_hash(array):
    array = np.asarray(array)
    sha = hashlib.sha256()
    sha.update(str(array.dtype).encode())
    sha.update(str(array.shape).encode())
    sha.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    return sha.hexdigest()


def pair(value):
    if type(value) is int:
        return (value,value)
    if isinstance(value,(list,tuple)) and len(value) == 2 and all(type(x) is int for x in value):
        return tuple(value)
    raise UnsupportedOperator(f"expected constant integer pair, got {value!r}")


def from_torch(model,example,name=None):
    if any(module.training for module in model.modules()):
        raise ValueError("inference import requires model.eval() including every submodule")
    if example.device.type != 'cpu' or example.dtype not in (torch.float32,torch.float64) or example.ndim not in (2,4):
        raise ValueError("import expects a CPU float32/float64 NF/NCHW example")
    if any(p.device.type != 'cpu' or p.dtype != example.dtype for p in model.parameters()):
        raise ValueError("model parameters must match CPU input floating-point precision")
    graph = symbolic_trace(model)
    with torch.no_grad():
        ShapeProp(graph).propagate(example)
    parameters,tensors,ops = {},[],[]
    placeholders = []
    outputs = []

    def reject(node,reason):
        raise UnsupportedOperator(f"FX node {node.name} ({node.op}: {node.target}): {reason}")

    def input_nodes(node):
        nodes = []
        def visit(value):
            if isinstance(value,FXNode): nodes.append(value)
            elif isinstance(value,(tuple,list)):
                for v in value: visit(v)
            elif isinstance(value,dict):
                for v in value.values(): visit(v)
        visit((node.args,node.kwargs))
        return tuple(n.name for n in nodes)

    def constant(node,index,key,default):
        return node.kwargs.get(key,node.args[index] if len(node.args) > index else default)

    def check_inplace(node,inplace):
        if inplace and len(node.args[0].users) != 1:
            reject(node,"in-place mutation of a tensor with other consumers is unsupported")

    def capture(op,module):
        if op.kind in ('Conv','Linear'):
            parameters[f'weights:{op.id}'] = module.weight.detach().numpy().copy()
            if op.bias: parameters[f'bias:{op.id}'] = module.bias.detach().numpy().copy()
        if op.kind == 'BatchNorm':
            if not module.track_running_stats or module.running_mean is None or module.running_var is None:
                raise UnsupportedOperator("BatchNorm inference requires fixed running statistics")
            parameters[f'mean:{op.id}'] = module.running_mean.detach().numpy().copy()
            parameters[f'variance:{op.id}'] = module.running_var.detach().numpy().copy()
            if op.affine:
                parameters[f'scale:{op.id}'] = module.weight.detach().numpy().copy()
                parameters[f'shift:{op.id}'] = module.bias.detach().numpy().copy()

    for node in graph.graph.nodes:
        if node.op == 'output':
            def output_ids(value):
                if isinstance(value,FXNode): return [value.name]
                if isinstance(value,(tuple,list)): return [v for child in value for v in output_ids(child)]
                reject(node,"only tensor/tuple/list outputs are supported")
            outputs = output_ids(node.args[0])
            continue
        metadata = node.meta.get('tensor_meta')
        if metadata is None or not hasattr(metadata,'shape'):
            reject(node,"non-tensor/control/constant node unsupported")
        if metadata.dtype != example.dtype:
            reject(node,"mixed precision unsupported")
        tensors.append(Tensor(node.name,tuple(metadata.shape),example.element_size()))
        if node.op == 'placeholder':
            placeholders.append(node.name)
            continue
        names = input_nodes(node)
        op = None
        if node.op == 'call_module':
            module = graph.get_submodule(node.target)
            common = dict(id=node.name,inputs=names,output=node.name)
            if isinstance(module,nn.Conv2d):
                if module.padding_mode != 'zeros': reject(node,"nonzero padding mode unsupported")
                op = Operator(kind='Conv',kernel=pair(module.kernel_size),stride=pair(module.stride),
                    padding=pair(module.padding),dilation=pair(module.dilation),groups=module.groups,
                    bias=module.bias is not None,**common)
            elif isinstance(module,nn.Linear):
                op = Operator(kind='Linear',bias=module.bias is not None,**common)
            elif isinstance(module,(nn.BatchNorm1d,nn.BatchNorm2d)):
                op = Operator(kind='BatchNorm',epsilon=module.eps,affine=module.affine,**common)
            elif isinstance(module,(nn.ReLU,nn.ReLU6)):
                check_inplace(node,module.inplace)
                op = Operator(kind='ReLU6' if isinstance(module,nn.ReLU6) else 'ReLU',**common)
            elif isinstance(module,nn.MaxPool2d):
                if module.return_indices: reject(node,"pool indices output unsupported")
                op = Operator(kind='MaxPool',kernel=pair(module.kernel_size),stride=pair(module.stride or module.kernel_size),
                    padding=pair(module.padding),dilation=pair(module.dilation),ceil_mode=module.ceil_mode,**common)
            elif isinstance(module,nn.AvgPool2d):
                op = Operator(kind='AvgPool',kernel=pair(module.kernel_size),stride=pair(module.stride or module.kernel_size),
                    padding=pair(module.padding),ceil_mode=module.ceil_mode,count_include_pad=module.count_include_pad,
                    divisor_override=module.divisor_override,**common)
            elif isinstance(module,nn.AdaptiveAvgPool2d):
                op = Operator(kind='AdaptiveAvgPool',**common)
            elif isinstance(module,nn.Flatten):
                rank = len(next(t.shape for t in tensors if t.id == names[0]))
                if module.start_dim != 1 or module.end_dim not in (-1,rank-1):
                    reject(node,"Flatten must flatten all nonbatch dimensions")
                op = Operator(kind='Flatten',**common)
            elif isinstance(module,(nn.Dropout,nn.Dropout2d,nn.Identity)):
                op = Operator(kind='Identity',**common)
            else:
                reject(node,f"unsupported module {type(module).__name__}")
            capture(op,module)
        elif node.op == 'call_function':
            common = dict(id=node.name,inputs=names,output=node.name)
            if node.target in (python_operator.add,torch.add):
                if constant(node,2,'alpha',1) != 1 or node.kwargs.get('out') is not None:
                    reject(node,"Add requires alpha=1 and no output mutation")
                op = Operator(kind='Add',**common)
            elif node.target == torch.flatten:
                rank = len(next(t.shape for t in tensors if t.id == names[0]))
                if constant(node,1,'start_dim',0) != 1 or constant(node,2,'end_dim',-1) not in (-1,rank-1):
                    reject(node,"Flatten must flatten all nonbatch dimensions")
                op = Operator(kind='Flatten',**common)
            elif node.target in (torch.relu,F.relu,F.relu6):
                check_inplace(node,node.kwargs.get('inplace',False))
                op = Operator(kind='ReLU6' if node.target == F.relu6 else 'ReLU',**common)
            elif node.target == F.adaptive_avg_pool2d:
                op = Operator(kind='AdaptiveAvgPool',**common)
            elif node.target in (F.max_pool2d,F.avg_pool2d):
                kernel = constant(node,1,'kernel_size',None)
                stride = constant(node,2,'stride',None) or kernel
                attrs = dict(kernel=pair(kernel),stride=pair(stride),padding=pair(constant(node,3,'padding',0)))
                if node.target == F.max_pool2d:
                    if constant(node,6,'return_indices',False): reject(node,"pool indices unsupported")
                    attrs.update(dilation=pair(constant(node,4,'dilation',1)),ceil_mode=constant(node,5,'ceil_mode',False))
                    kind = 'MaxPool'
                else:
                    attrs.update(ceil_mode=constant(node,4,'ceil_mode',False),count_include_pad=constant(node,5,'count_include_pad',True),
                                 divisor_override=constant(node,6,'divisor_override',None))
                    kind = 'AvgPool'
                op = Operator(kind=kind,**attrs,**common)
            else:
                reject(node,"unsupported function; no fallback or silent omission")
        elif node.op == 'call_method' and node.target == 'flatten':
            rank = len(next(t.shape for t in tensors if t.id == names[0]))
            if constant(node,1,'start_dim',0) != 1 or constant(node,2,'end_dim',-1) not in (-1,rank-1):
                reject(node,"Flatten must flatten all nonbatch dimensions")
            op = Operator(node.name,'Flatten',names,node.name)
        else:
            reject(node,"unsupported method or graph node")
        ops.append(op)
    if len(placeholders) != 1:
        raise UnsupportedOperator("exactly one tensor model input is currently supported")
    manifest = {key:{'shape':list(a.shape),'dtype':str(a.dtype),'bytes':a.nbytes,'sha256':array_hash(a)}
                for key,a in sorted(parameters.items())}
    state_manifest = {key:array_hash(t.detach().numpy()) for key,t in model.state_dict().items()}
    metadata = (('torch_version',str(torch.__version__)),('graph_sha256',hashlib.sha256(graph.code.encode()).hexdigest()),
                ('model_state_sha256',digest(state_manifest)),('parameter_manifest_sha256',digest(manifest)),
                ('precision',str(example.dtype)),('semantics','eval; explicit operators; no fusion'))
    workload = Workload(tuple(tensors),tuple(ops),tuple(outputs),name or type(model).__name__,metadata)
    workload.validate()
    if set(workload.parameter_map) != set(parameters):
        raise ValueError("inference parameter extraction disagrees with IR")
    return ImportedModel(workload,parameters,manifest,graph,placeholders[0])


def make_torchvision_model(name,seed=7):
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"model must be one of {SUPPORTED_MODELS}")
    import torchvision.models as models
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = getattr(models,name)(weights=None)
    return model.eval()
