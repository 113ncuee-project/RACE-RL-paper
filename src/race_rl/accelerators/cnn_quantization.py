"""Explicit seeded per-tensor INT8 calibration and FP32 postprocessing reference.

This is a numerical protocol, not pretrained model accuracy validation. Random
Torchvision weights and one seeded calibration input do not establish accuracy.
"""
from dataclasses import dataclass,replace
import math
import numpy as np
import torch
from torch.fx import Interpreter
import torch.nn.functional as F
from ..serialization import digest
from ..frontend import array_hash
from .gemmini_model import execute_gemm,GemminiAction
from .cnn_design import output_tiles


def scale_of(value):
    x=np.asarray(value)
    if not np.isfinite(x).all():raise ValueError('nonfinite calibration tensor')
    # Scale metadata is physically stored as FP32, not an undeclared FP64
    # scalar that would change products/reciprocals during hardware execution.
    return float(np.float32(max(float(np.max(np.abs(x))),1e-12)/127))


def quantize(value,scale):
    if not math.isfinite(scale) or scale<=0:raise ValueError('positive quantization scale required')
    x=np.asarray(value,dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError('nonfinite requantization value')
    return np.clip(np.rint(x*np.float32(1/scale)),-127,127).astype(np.int8)


@dataclass
class QuantizedCNN:
    workload: object
    parameters: dict
    scales: dict
    weight_scales: dict
    manifest: dict
    source_output: np.ndarray


def calibrate(imported,example):
    scales={}
    class Capture(Interpreter):
        def run_node(self,node):
            value=super().run_node(node)
            if node.name in imported.workload.tensor_map:
                scales[node.name]=scale_of(value.detach().numpy())
            return value
    with torch.no_grad():source=Capture(imported.graph_module).run(example)
    if isinstance(source,(tuple,list)):
        if len(source)!=1:raise ValueError('CNN runner currently requires one output')
        source=source[0]
    parameters={};weight_scales={}
    for name,value in imported.parameters.items():
        if name.startswith('weights:'):
            weight_scales[name]=scale_of(value)
            parameters[name]=quantize(value,weight_scales[name])
        else:parameters[name]=np.asarray(value,dtype=np.float32)
    aliases={o.output:o.inputs[0] for o in imported.workload.operators if o.kind in ('Flatten','Identity')}
    for output,parent in aliases.items():scales[output]=scales[parent]
    manifest={'activation_scales':scales,'weight_scales':weight_scales,
        'parameters':{name:{'shape':list(v.shape),'dtype':str(v.dtype),'bytes':v.nbytes,'sha256':array_hash(v)} for name,v in parameters.items()},
        'protocol':'symmetric per-tensor signed INT8 [-127,127]; FP32 reciprocal; nearest-even; saturation; INT32 MAC, FP32 postops',
        'calibration':'one seeded input; source weights are seeded random initialization; no accuracy/representativeness claim',
        'accuracy_validated':False,'source_parameter_manifest':imported.parameter_manifest}
    workload=replace(imported.workload,tensors=tuple(replace(t,bytes_per_element=1) for t in imported.workload.tensors),
        source_metadata=(*imported.workload.source_metadata,('quantization_sha256',digest(manifest)),('hardware_precision','INT8/INT32 plus FP32 vector')))
    return QuantizedCNN(workload,parameters,scales,weight_scales,manifest,source.detach().numpy().copy())


def integer_conv_reference(x,weights,op):
    """Integer-valued FP64 kernel: every product/sum exact within INT32 bounds.

    Used for full-model reference speed only. Tiny independent INT64 loops test
    grouped/depthwise/dilated cases, so this is not the timing evaluator.
    """
    worst=x.shape[1]//op.groups*math.prod(op.kernel)*int(np.max(np.abs(x.astype(np.int16))))*int(np.max(np.abs(weights.astype(np.int16))))
    if worst>=2**31:raise ValueError('INT32 partial-sum range cannot be guaranteed')
    result=F.conv2d(torch.from_numpy(x.astype(np.float64)),torch.from_numpy(weights.astype(np.float64)),
                    stride=op.stride,padding=op.padding,dilation=op.dilation,groups=op.groups).numpy()
    if result.size and (np.max(np.abs(result))>2**31-1 or not np.array_equal(result,np.rint(result))):
        raise ValueError('INT32 overflow/noninteger convolution reference')
    return result.astype(np.int32)


def tiled_integer_conv(workload,op,x,weights,plan):
    """Small independent H/W/C gather + K16 GEMM reconstruction reference.

    This is for verification fixtures, not run inside action selection or every
    hardware evaluation. It covers grouped/dilated/strided halos and tails.
    """
    result=np.zeros(workload.tensor_map[op.output].shape,dtype=np.int32)
    groupin=x.shape[1]//op.groups;groupout=result.shape[1]//op.groups
    for tile,_ in output_tiles(workload,op,plan,1):
        nb,channels,ys,xs=tile.output.bounds
        begin,end=channels
        while begin<end:
            group=begin//groupout;stop=min(end,(group+1)*groupout)
            positions=[(n,y,z) for n in range(*nb) for y in range(*ys) for z in range(*xs)]
            a=np.zeros((len(positions),groupin*math.prod(op.kernel)),dtype=np.int8)
            for row,(n,y,z) in enumerate(positions):
                col=0
                for ci in range(groupin):
                    for ky in range(op.kernel[0]):
                        for kx in range(op.kernel[1]):
                            iy=y*op.stride[0]-op.padding[0]+ky*op.dilation[0]
                            ix=z*op.stride[1]-op.padding[1]+kx*op.dilation[1]
                            if 0<=iy<x.shape[2] and 0<=ix<x.shape[3]:a[row,col]=x[n,group*groupin+ci,iy,ix]
                            col+=1
            b=weights[begin:stop].reshape(stop-begin,-1).T
            values=execute_gemm(a,b,GemminiAction())
            for row,(n,y,z) in enumerate(positions):result[n,begin:stop,y,z]=values[row]
            begin=stop
    return result


def execute_quantized(bundle,example):
    w,s,params=bundle.workload,bundle.scales,bundle.parameters
    external=[name for name in w.tensor_map if name not in w.producers]
    if len(external)!=1:raise ValueError('one external tensor required')
    values={external[0]:quantize(example,s[external[0]])};clipped=0;count=0
    remaining={name:sum(name in op.inputs for op in w.operators) for name in w.tensor_map}
    for op in w.operators:
        inputs=[values[name] for name in op.inputs]
        real=[v.astype(np.float32)*np.float32(s[name]) for name,v in zip(op.inputs,inputs)]
        if op.kind in ('Identity','Flatten'):
            result=inputs[0].reshape(w.tensor_map[op.output].shape)
        else:
            if op.kind in ('Conv','Linear'):
                weights=params['weights:'+op.id]
                if op.kind=='Conv':integer=integer_conv_reference(inputs[0],weights,op)
                else:
                    integer=inputs[0].astype(np.int64)@weights.astype(np.int64).T
                    if integer.size and (integer.min()<-(2**31) or integer.max()>=2**31):raise ValueError('INT32 Linear overflow')
                out=integer.astype(np.float32)*np.float32(s[op.inputs[0]]*bundle.weight_scales['weights:'+op.id])
                if op.bias:
                    bias=params['bias:'+op.id]
                    out=out+bias.reshape(1,-1,1,1) if out.ndim==4 else out+bias
            elif op.kind=='Add':out=real[0]+real[1]
            elif op.kind in ('ReLU','ReLU6'):out=np.clip(real[0],0,6 if op.kind=='ReLU6' else np.inf)
            elif op.kind=='BatchNorm':
                shape=(1,-1,1,1) if real[0].ndim==4 else (1,-1)
                out=(real[0]-params['mean:'+op.id].reshape(shape))/np.sqrt(params['variance:'+op.id].reshape(shape)+np.float32(op.epsilon))
                if op.affine:out=out*params['scale:'+op.id].reshape(shape)+params['shift:'+op.id].reshape(shape)
            elif op.kind=='MaxPool':out=F.max_pool2d(torch.from_numpy(real[0]),op.kernel,op.stride,op.padding,op.dilation,op.ceil_mode).numpy()
            elif op.kind=='AvgPool':out=F.avg_pool2d(torch.from_numpy(real[0]),op.kernel,op.stride,op.padding,op.ceil_mode,op.count_include_pad,op.divisor_override).numpy()
            elif op.kind=='AdaptiveAvgPool':out=F.adaptive_avg_pool2d(torch.from_numpy(real[0]),w.tensor_map[op.output].shape[2:]).numpy()
            else:raise ValueError('unsupported quantized operator '+op.kind)
            ratio=np.asarray(out,dtype=np.float32)*np.float32(1/s[op.output])
            clipped+=int(np.count_nonzero(np.abs(ratio)>127));count+=ratio.size
            result=quantize(out,s[op.output])
        if result.shape!=w.tensor_map[op.output].shape:raise ValueError('quantized output shape mismatch')
        values[op.output]=result
        for name in set(op.inputs):
            remaining[name]-=1
            if remaining[name]==0 and name not in w.outputs:values.pop(name,None)
    output=values[w.outputs[0]]
    dequant=output.astype(np.float32)*np.float32(s[w.outputs[0]])
    reference=bundle.source_output.astype(np.float32)
    error=dequant-reference
    report={'status':'EXECUTED_WITH_EXPLICIT_QUANTIZATION','operator_count':len(w.operators),
            'int8_output_sha256':array_hash(output),'dequantized_max_abs_error':float(np.max(np.abs(error))),
            'dequantized_rmse':float(np.sqrt(np.mean(error.astype(np.float64)**2))),
            'saturated_values':clipped,'requantized_values':count,'accuracy_validated':False,
            'interpretation':'numerical execution/overflow/shape checks; errors versus seeded FP32 network, not dataset accuracy or proof of acceptable quantization'}
    return output,report
