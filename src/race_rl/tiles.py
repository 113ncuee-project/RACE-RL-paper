"""Output ownership, grouped/dilated halos and exact Flatten coordinate transformation."""
from dataclasses import dataclass
from math import prod, ceil
from .workload import Workload


@dataclass(frozen=True)
class Region:
    bounds: tuple[tuple[int, int], ...]

    def __post_init__(self):
        if not self.bounds or any(type(a) is not int or type(b) is not int or a < 0 or b <= a
                                  for a,b in self.bounds):
            raise ValueError("invalid region")

    @property
    def elements(self):
        return prod(b-a for a,b in self.bounds)

    def intersection(self, other):
        if len(self.bounds) != len(other.bounds):
            raise ValueError("region rank mismatch")
        spans = tuple((max(a,c),min(b,d)) for (a,b),(c,d) in zip(self.bounds,other.bounds))
        return Region(spans) if all(a < b for a,b in spans) else None


@dataclass(frozen=True)
class Tile:
    id: str
    operator: str
    output: Region


def whole(shape):
    return Region(tuple((0,s) for s in shape))


def partition(workload: Workload, counts=None):
    workload.validate()
    counts = counts or {}
    if not set(counts) <= set(workload.operator_map):
        raise ValueError("unknown partition operator")
    result = []
    for op in workload.operators:
        shape = workload.tensor_map[op.output].shape
        axis = 2 if len(shape) == 4 else 1
        count = counts.get(op.id,1)
        if type(count) is not int or not 1 <= count <= shape[axis]:
            raise ValueError("invalid tile count")
        for i in range(count):
            bounds = list(whole(shape).bounds)
            bounds[axis] = (i*shape[axis]//count,(i+1)*shape[axis]//count)
            result.append(Tile(f"{op.id}:{i}",op.id,Region(tuple(bounds))))
    validate_coverage(workload,result)
    return tuple(result)


def validate_coverage(workload, tiles):
    if len({t.id for t in tiles}) != len(tiles):
        raise ValueError("duplicate tiles")
    if any(t.operator not in workload.operator_map for t in tiles):
        raise ValueError("unknown tile operator")
    for op in workload.operators:
        target = whole(workload.tensor_map[op.output].shape)
        group = [t for t in tiles if t.operator == op.id]
        if sum(t.output.elements for t in group) != target.elements:
            raise ValueError("incomplete output coverage")
        for i,t in enumerate(group):
            if target.intersection(t.output) != t.output:
                raise ValueError("output out of bounds")
            if any(t.output.intersection(u.output) is not None for u in group[i+1:]):
                raise ValueError("overlapping output ownership")
    return True


def unflatten_regions(shape, batch, features):
    if len(shape) == 2:
        return (Region((batch,features)),)
    channels,height,width = shape[1:]
    cursor,end = features
    result = []
    while cursor < end:
        channel,rem = divmod(cursor,height*width)
        row,column = divmod(rem,width)
        if rem == 0 and end-cursor >= height*width:
            count = (end-cursor)//(height*width)
            result.append(Region((batch,(channel,channel+count),(0,height),(0,width))))
            cursor += count*height*width
        elif column == 0 and end-cursor >= width:
            count = min(height-row,(end-cursor)//width)
            result.append(Region((batch,(channel,channel+1),(row,row+count),(0,width))))
            cursor += count*width
        else:
            count = min(width-column,end-cursor)
            result.append(Region((batch,(channel,channel+1),(row,row+1),(column,column+count))))
            cursor += count
    return tuple(result)


def requirements(workload, tile):
    op = workload.operator_map[tile.operator]
    out = tile.output
    result = []
    for name in op.inputs:
        shape = workload.tensor_map[name].shape
        if op.kind in {"Add","ReLU","ReLU6","BatchNorm","Identity"}:
            regions = (out,)
        elif op.kind == "Flatten":
            regions = unflatten_regions(shape,out.bounds[0],out.bounds[1])
        elif op.kind == "Linear":
            regions = (Region((out.bounds[0],(0,shape[1]))),)
        elif op.kind == "AdaptiveAvgPool":
            target = workload.tensor_map[op.output].shape
            spatial = tuple((a*n//m,ceil(b*n/m)) for (a,b),n,m in zip(out.bounds[2:],shape[2:],target[2:]))
            regions = (Region((out.bounds[0],out.bounds[1],*spatial)),)
        else:
            spatial = tuple((max(0,a*s-p),min(n,(b-1)*s-p+d*(k-1)+1)) for (a,b),s,p,d,k,n in
                            zip(out.bounds[2:],op.stride,op.padding,op.dilation,op.kernel,shape[2:]))
            if any(a >= b for a,b in spatial):
                raise ValueError("all-padding windows unsupported")
            if op.kind == "Conv":
                output_channels = workload.tensor_map[op.output].shape[1]
                group_out,group_in = output_channels//op.groups,shape[1]//op.groups
                a,b = out.bounds[1]
                channels = ((a//group_out)*group_in,((b-1)//group_out+1)*group_in)
            else:
                channels = out.bounds[1]
            regions = (Region((out.bounds[0],channels,*spatial)),)
        result.extend((name,r) for r in regions)
    return tuple(result)


def macs(workload, tile):
    op = workload.operator_map[tile.operator]
    shape = workload.tensor_map[op.inputs[0]].shape
    if op.kind == "Conv":
        return tile.output.elements*(shape[1]//op.groups)*prod(op.kernel)
    if op.kind == "Linear":
        return tile.output.elements*shape[1]
    return 0


def operations(workload, tile):
    op = workload.operator_map[tile.operator]
    outputs = tile.output.elements
    if op.kind in {"Conv","Linear"}:
        return macs(workload,tile)+(outputs if op.bias else 0)
    if op.kind in {"AvgPool","MaxPool"}:
        return outputs*prod(op.kernel)
    if op.kind == "AdaptiveAvgPool":
        src = workload.tensor_map[op.inputs[0]].shape
        dst = workload.tensor_map[op.output].shape
        spans = [sum(ceil((i+1)*n/m)-i*n//m for i in range(a,b))
                 for (a,b),n,m in zip(tile.output.bounds[2:],src[2:],dst[2:])]
        return prod(spans)*prod(b-a for a,b in tile.output.bounds[:2])+outputs
    if op.kind == "BatchNorm":
        channels = tile.output.bounds[1][1]-tile.output.bounds[1][0]
        return outputs*(4 if op.affine else 2)+channels*2
    if op.kind in {"Identity","Flatten"}:
        return 0
    return outputs*(2 if op.kind == "ReLU6" else 1)


def parameter_regions(workload, tile):
    op = workload.operator_map[tile.operator]
    return tuple((name,Region((tile.output.bounds[1],*whole(shape).bounds[1:])))
                 for name,shape in workload.parameter_shapes(op).items())


def weight_bytes(workload, tile):
    return sum(r.elements*workload.tensor_map[workload.operator_map[tile.operator].output].bytes_per_element
               for name,r in parameter_regions(workload,tile) if name.startswith("weights:"))
