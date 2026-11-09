"""Whole-CNN analytical hierarchy model with explicit source ownership/liveness.

Layer barriers, streamed gather/packing, and a declared FP32 vector unit are
research assumptions. Timing is a bottleneck service estimate, not an RTL
schedule or an upper-bound guarantee. No old synthetic oracle is used.
"""
from dataclasses import asdict
from collections import defaultdict,Counter
import math
import heapq
from ..serialization import digest
from ..tiles import requirements,whole,Region,unflatten_regions,operations,macs,parameter_regions
from ..mesh import mesh_shape
from .gemmini_config import default_config
from .gemmini_model import PlatformScenario,xy_route,padded,ceildiv
from .memory_hierarchy import MemoryHierarchy
from .cnn_design import CNNDesign,output_tiles
from .cnn_kernel import kernel_counts


def axis_windows(lo,hi,outlo,outhi,kernel,stride,pad,dilation):
    """Return selected output count and physical contiguous input span per tap."""
    result=[]
    for tap in range(kernel):
        begin=max(outlo,ceildiv(lo+pad-tap*dilation,stride))
        end=min(outhi,(hi-1+pad-tap*dilation)//stride+1)
        if end>begin:result.append((end-begin,(end-begin-1)*stride+1))
    return result


def input_volume(workload,op,tile,source_region):
    """Physical row-span payload and useful references; no free gather cache."""
    out=tile.output.bounds;bounds=source_region.bounds
    if op.kind in ('Conv','MaxPool','AvgPool'):
        ys=axis_windows(*bounds[2],*out[2],op.kernel[0],op.stride[0],op.padding[0],op.dilation[0])
        xs=axis_windows(*bounds[3],*out[3],op.kernel[1],op.stride[1],op.padding[1],op.dilation[1])
        bc=(bounds[0][1]-bounds[0][0])*(bounds[1][1]-bounds[1][0])
        useful=bc*sum(v[0] for v in ys)*sum(v[0] for v in xs)
        payload=bc*sum(v[0] for v in ys)*sum(v[1] for v in xs)
        # Rows are 16-byte aligned. A worst-offset bound is explicit; no
        # coalescing across unrelated kernel taps/channels is assumed.
        rows=bc*sum(v[0] for v in ys)
        transactions=rows*sum(ceildiv(span+15,64) for _,span in xs)
        beats=rows*sum(ceildiv(span+15,16) for _,span in xs)
        return payload,useful,transactions,beats
    if op.kind=='AdaptiveAvgPool':
        source=workload.tensor_map[op.inputs[0]].shape
        target=workload.tensor_map[op.output].shape
        spans=[]
        for axis in (2,3):
            sizes=[]
            for i in range(*out[axis]):
                a=max(bounds[axis][0],i*source[axis]//target[axis])
                b=min(bounds[axis][1],ceildiv((i+1)*source[axis],target[axis]))
                if b>a:sizes.append(b-a)
            spans.append(sizes)
        bc=(bounds[0][1]-bounds[0][0])*(bounds[1][1]-bounds[1][0])
        amount=bc*sum(spans[0])*sum(spans[1])
        rows=bc*sum(spans[0])
        return amount,amount,rows*sum(ceildiv(x+15,64) for x in spans[1]),rows*sum(ceildiv(x+15,16) for x in spans[1])
    width=bounds[-1][1]-bounds[-1][0]
    rows=source_region.elements//width
    return source_region.elements,source_region.elements,rows*ceildiv(width+15,64),rows*ceildiv(width+15,16)


class GemminiCNNModel:
    vector_lanes=16
    vector_ops_per_cycle=16
    activation_cache_bytes=64*1024 # GB bank 0; bank 1 is gather/A/output staging.

    def __init__(self,workload,*,memory=None,platform=None):
        workload.validate()
        self.workload=workload
        self.memory=memory or MemoryHierarchy()
        self.platform=platform or PlatformScenario()
        self.config=default_config()
        self.op_index={o.id:i for i,o in enumerate(workload.operators)}
        self.aliases={o.output:o.inputs[0] for o in workload.operators if o.kind in ('Flatten','Identity')}
        def root(name):
            while name in self.aliases:name=self.aliases[name]
            return name
        self.roots={name:root(name) for name in workload.tensor_map}
        self.last_use=defaultdict(lambda:-1)
        for i,op in enumerate(workload.operators):
            for name in op.inputs:self.last_use[root(name)]=max(i,self.last_use[root(name)])
        for name in workload.outputs:self.last_use[root(name)]=len(workload.operators)

    def _query(self,name,region,layouts):
        if name in self.aliases:
            parent=self.aliases[name]
            shape=self.workload.tensor_map[parent].shape
            transformed=unflatten_regions(shape,*region.bounds) if len(region.bounds)==2 and len(shape)==4 else (region,)
            for r in transformed:yield from self._query(parent,r,layouts)
        else:
            records,index=layouts[name]
            # Channel bins avoid testing every spatial tile against every query.
            first,last=region.bounds[1]
            candidates={i for channel in range(first,last) for i in index.get(channel,())}
            for i in sorted(candidates):
                previous,location=records[i]
                overlap=previous.intersection(region)
                if overlap is not None:yield overlap,location

    def evaluate(self,design):
        if not isinstance(design,CNNDesign) or len(design.plans)!=len(self.workload.operators):
            raise ValueError('one complete plan per CNN operator required')
        w,c,h,p=self.workload,self.config,self.memory,self.platform
        count=design.compute_count
        # Validate only the already selected complete action. These are
        # physical bounds, not actor-side PPA filtering or candidate trials.
        for op,plan in zip(w.operators,design.plans):
            if op.kind in ('Flatten','Identity'):continue
            if plan.keep_output and w.tensor_map[op.output].bytes>count*self.activation_cache_bytes:
                raise ValueError('CNN GB activation live set exceeds total reserved cache capacity')
            if op.kind in ('Conv','Linear') and plan.keep_weights:
                source=w.tensor_map[op.inputs[0]].shape
                k=source[1] if op.kind=='Linear' else source[1]//op.groups*math.prod(op.kernel)
                n=w.tensor_map[op.output].shape[1]//(op.groups if op.kind=='Conv' else 1)
                if padded(k)*padded(min(n,plan.c))>2*c.scratchpad_bank_bytes:
                    raise ValueError('CNN resident weights exceed GB banks 2/3')
        layouts={};live_bytes=[0]*count;live_objects={};peak=[0]*count
        traffic={'external_bytes':0,'dma_bus_bytes':0,'dma_transactions':0,'directed_link_byte_hops':0,
                 'by_operand':{'activation':0,'weights':0,'constants':0,'output':0},
                 'storage_access_bytes':{key:0 for key in ('gb','lb','rf','accumulator')},
                 'hierarchy_movement_bytes':defaultdict(int)}
        links=defaultdict(int);link_flows=defaultdict(int);attribution=defaultdict(int);rows=[];placements=[];largest=[]
        totals={'logical_macs':0,'padded_macs':0,'vector_ops':0,'gather_references':0}
        dram_allocation=0;cycles=0;kernel_cache={};flow_serial=0
        for name,tensor in w.tensor_map.items():
            if name not in w.producers:
                records=[(whole(tensor.shape),'dram')]
                layouts[name]=(records,{ch:[0] for ch in range(tensor.shape[1])})
                dram_allocation+=math.prod(tensor.shape[:-1])*padded(tensor.shape[-1])
        for op in w.operators:
            if op.kind=='Conv':
                cin=w.tensor_map[op.inputs[0]].shape[1]//op.groups
                cout=w.tensor_map[op.output].shape[1]//op.groups
                dram_allocation+=op.groups*padded(cin*math.prod(op.kernel))*padded(cout)
            elif op.kind=='Linear':
                dram_allocation+=padded(w.tensor_map[op.inputs[0]].shape[1])*padded(w.tensor_map[op.output].shape[1])
            dram_allocation+=sum(math.prod(shape)*4 for name,shape in w.parameter_shapes(op).items() if not name.startswith('weights:'))+12 # tensor scale metadata
        for index,(op,plan) in enumerate(zip(w.operators,design.plans)):
            if op.output in self.aliases:
                rows.append({'operator':op.id,'kind':op.kind,'cycles':0,'alias_of':op.inputs[0],
                             'logical_macs':0,'tile_count':0,'placements':{}})
                continue
            tiles=list(output_tiles(w,op,plan,count))
            bynode=Counter(node for _,node in tiles)
            placements.append({'operator':op.id,'tiles_by_node':{f'c{node}':n for node,n in bynode.items()},
                               'profile':asdict(plan),'tile_count':len(tiles)})
            # Every tensor output has a physical padded-row allocation in DRAM
            # even if this action retains it on-chip. No implicit recompression.
            dram_allocation+=sum(math.prod(b-a for a,b in t.output.bounds[:-1])*padded(t.output.bounds[-1][1]-t.output.bounds[-1][0]) for t,_ in tiles)
            if dram_allocation>p.dram_capacity_bytes:raise ValueError('CNN external static allocation exceeds DRAM capacity')
            node_service=[0]*count;external_cycles=0;link_cycles=defaultdict(int)
            weight_cache={};opmac=oppadded=opvec=0;opbytes=0
            records=[];channel_index=defaultdict(list);output_sizes=[0]*count
            def add_access(level,amount):traffic['storage_access_bytes'][level]+=amount
            def move(a,b,amount):traffic['hierarchy_movement_bytes'][a+'->'+b]+=amount
            def transfer(source,node,amount,transactions,beats,operand,write=False,dependency=None):
                nonlocal external_cycles,opbytes,flow_serial
                opbytes+=amount
                if source=='dram':
                    traffic['external_bytes']+=amount;traffic['by_operand'][operand]+=amount
                    traffic['dma_transactions']+=transactions;traffic['dma_bus_bytes']+=beats*16
                    external_cycles+=ceildiv(beats*16,min(p.dram_bytes_per_cycle,c.dma_beat_bytes,h.gb_bank_bytes_per_cycle))+transactions*p.dram_first_response_cycles
                    origin,destination=(node,0) if write else (0,node)
                    move('gb','dram',amount) if write else move('dram','gb',amount)
                    add_access('gb',amount)
                else:
                    origin,destination=source,node
                    add_access('gb',amount) # source cache read, including local gather
                    if origin!=destination:add_access('gb',amount);move('gb','gb',amount)
                route=xy_route(count,origin,destination)
                for a,b in route:
                    links[a,b]+=amount
                    if amount:link_flows[a,b]+=1
                    if dependency and dependency[0]:attribution[(*dependency,a,b,'read')]+=amount
                    if write:
                        for consumer in w.operators:
                            if op.output in consumer.inputs:
                                attribution[op.id,consumer.id,a,b,'spill_write']+=amount
                    link_cycles[a,b]+=ceildiv(amount,min(p.mesh_bytes_per_cycle,h.gb_bank_bytes_per_cycle))+p.mesh_hop_cycles
                traffic['directed_link_byte_hops']+=amount*len(route)
                if amount:
                    row={'source':'dram0' if source=='dram' and not write else f'c{origin}',
                                    'destination':'dram0' if write else f'c{node}',
                                    'tensor':op.output if write else operand+':'+op.id,'bytes':amount,
                                    'route':[{'from':f'c{a}','to':f'c{b}','link':f'c{min(a,b)}-c{max(a,b)}'} for a,b in route]}
                    flow_serial+=1
                    heapq.heappush(largest,(amount,flow_serial,row))
                    if len(largest)>30:heapq.heappop(largest)
            for tile,node in tiles:
                input_refs=0
                for name,needed in requirements(w,tile):
                    for region,source in self._query(name,needed,layouts):
                        if name in self.aliases:
                            # Flatten is a view; the requested source region is
                            # already transformed into physical NCHW spans.
                            width=region.bounds[-1][1]-region.bounds[-1][0]
                            amount=region.elements;nr=amount//width
                            volume=(amount,amount,nr*ceildiv(width+15,64),nr*ceildiv(width+15,16))
                        else:volume=input_volume(w,op,tile,region)
                        amount,refs,tr,beats=volume
                        transfer(source,node,amount,tr,beats,'activation',dependency=(w.producers.get(name),op.id))
                        # Remote/DRAM rows land in a 64-byte staging FIFO in GB
                        # bank 1. Local cache rows go directly to the packer.
                        if source=='dram' or source!=node:add_access('gb',amount)
                        node_service[node]+=ceildiv(amount,h.gb_bank_bytes_per_cycle)
                        if source!='dram' and source!=node:node_service[source]+=ceildiv(amount,h.gb_bank_bytes_per_cycle)
                        input_refs+=refs
                size=tile.output.elements
                totals['gather_references']+=input_refs
                if op.kind in ('Conv','Linear'):
                    bounds=tile.output.bounds
                    m=math.prod(b-a for a,b in bounds[2:]) if len(bounds)==4 else bounds[0][1]-bounds[0][0]
                    if op.kind=='Linear':segments=[(0,bounds[1][0],bounds[1][1],w.tensor_map[op.inputs[0]].shape[1])]
                    else:
                        groupout=w.tensor_map[op.output].shape[1]//op.groups
                        groupin=w.tensor_map[op.inputs[0]].shape[1]//op.groups
                        segments=[]
                        begin,end=bounds[1]
                        while begin<end:
                            group=begin//groupout;stop=min(end,(group+1)*groupout)
                            segments.append((group,begin,stop,groupin*math.prod(op.kernel)));begin=stop
                    for group,begin,end,k in segments:
                        key=(m,end-begin,k,plan.k)
                        if key not in kernel_cache:kernel_cache[key]=kernel_counts(*key,h,c)
                        cell=kernel_cache[key]
                        if cell['working_set']['gb_a_bytes']>c.scratchpad_bank_bytes:
                            raise ValueError('CNN gather FIFO/A panel exceeds GB staging bank')
                        if plan.keep_weights and cell['b_resident_bytes']>2*c.scratchpad_bank_bytes:
                            raise ValueError('CNN resident weights exceed GB banks 2/3')
                        bkey=(group,begin,end)
                        bbytes=0 if plan.keep_weights and weight_cache.get(node)==bkey else cell['b_resident_bytes'] if plan.keep_weights else cell['b_panel_bytes']
                        weight_cache[node]=bkey # one resident N-block per chiplet; replaces previous weights
                        if bbytes:
                            offset=begin-group*groupout if op.kind=='Conv' else begin
                            if offset%16:
                                # An odd-sized group may begin mid packed row.
                                # Fetch enclosing columns, then cost the B
                                # gather/zero-pad rather than assume aligned DMA.
                                source_bytes=padded(k)*padded(offset%16+end-begin)*(1 if plan.keep_weights else ceildiv(m,32))
                                transfer('dram',node,source_bytes,ceildiv(source_bytes,64),ceildiv(source_bytes,16),'weights')
                                packing=source_bytes+bbytes
                                add_access('gb',packing);add_access('lb',2*packing);add_access('rf',packing+12*bbytes)
                                move('gb','lb',source_bytes);move('lb','rf',source_bytes)
                                move('rf','lb',bbytes);move('lb','gb',bbytes)
                                opvec+=bbytes
                                node_service[node]+=ceildiv(packing,h.lb_bank_bytes_per_cycle)+ceildiv(bbytes,self.vector_ops_per_cycle)
                            else:transfer('dram',node,bbytes,ceildiv(bbytes,64),ceildiv(bbytes,16),'weights')
                        for level,amount in cell['access'].items():add_access(level,amount)
                        for boundary,amount in cell['moves'].items():traffic['hierarchy_movement_bytes'][boundary]+=amount
                        # Packer creates A in GB, including zeros. No packed A
                        # materialization in DRAM or free CPU preprocessing.
                        packing_ops=cell['a_packed_bytes']+m*(end-begin)*3
                        opvec+=packing_ops
                        add_access('rf',12*packing_ops)
                        node_service[node]+=cell['cycles']+ceildiv(packing_ops,self.vector_ops_per_cycle)
                        add_access('lb',2*cell['c_int32_bytes'])
                        add_access('rf',2*cell['c_int32_bytes'])
                        move('gb','lb',cell['c_int32_bytes']);move('lb','rf',cell['c_int32_bytes'])
                        opmac+=cell['logical_macs'];oppadded+=cell['padded_macs']
                    vec=size*(1 if op.bias else 0)
                else:
                    vec=operations(w,tile)+input_refs+3*size # dequantize, operator, quantize
                    # Vector chunks use LB/RF with 32-bit intermediates. They
                    # do not execute on the systolic array or bypass storage.
                    add_access('lb',2*input_refs+8*size)
                    add_access('rf',2*input_refs+8*size)
                    move('gb','lb',input_refs);move('lb','rf',input_refs)
                    move('rf','lb',4*size);move('lb','gb',4*size)
                    add_access('gb',4*size) # vector INT32/FP32 staging output
                    add_access('gb',4*size) # explicit reload before requantization
                    add_access('lb',8*size);add_access('rf',8*size)
                    move('gb','lb',4*size);move('lb','rf',4*size)
                    node_service[node]+=ceildiv(8*size,h.lb_bank_bytes_per_cycle)
                    node_service[node]+=ceildiv(2*input_refs+8*size,h.lb_bank_bytes_per_cycle)
                # FP32 post-op parameters are explicit, not omitted because
                # model activations/weights use INT8.
                constants=sum(r.elements*4 for name,r in parameter_regions(w,tile) if not name.startswith('weights:'))
                constants+=12
                if constants:
                    transfer('dram',node,constants,ceildiv(constants,64),ceildiv(constants,16),'constants')
                    add_access('gb',constants);add_access('lb',2*constants);add_access('rf',2*constants)
                    move('gb','lb',constants);move('lb','rf',constants)
                opvec+=vec;node_service[node]+=ceildiv(vec,self.vector_ops_per_cycle)
                add_access('gb',size);add_access('rf',12*vec+size);add_access('lb',2*size)
                node_service[node]+=ceildiv(2*size,h.lb_bank_bytes_per_cycle)
                move('rf','lb',size);move('lb','gb',size)
                width=tile.output.bounds[-1][1]-tile.output.bounds[-1][0]
                nr=size//width
                if not plan.keep_output:transfer('dram',node,size,nr*ceildiv(width+15,64),nr*ceildiv(width+15,16),'output',True)
                else:output_sizes[node]+=size
                records.append((tile.output,node if plan.keep_output else 'dram'))
                for channel in range(*tile.output.bounds[1]):channel_index[channel].append(len(records)-1)
            for node,amount in enumerate(output_sizes):
                # Conservative layer-barrier live set: input caches cannot be
                # reclaimed until all consumers in the layer have read them.
                if live_bytes[node]+amount>self.activation_cache_bytes:
                    raise ValueError('CNN GB activation live set exceeds reserved 64 KiB bank; choose DRAM residency or another layout')
                peak[node]=max(peak[node],live_bytes[node]+amount)
                live_bytes[node]+=amount
            if plan.keep_output:live_objects[op.output]=output_sizes
            layouts[op.output]=(records,channel_index)
            for name in list(live_objects):
                if self.last_use[name]<=index:
                    for node,amount in enumerate(live_objects.pop(name)):live_bytes[node]-=amount
            for name in list(layouts):
                if self.last_use[name]<=index and name!=op.output:
                    del layouts[name]
            # Bandwidth pressure is a measured counter / declared port rate,
            # not a hidden schedule. Layer pipeline/fill is not RTL calibrated.
            duration=max(max(node_service,default=0),external_cycles,max(link_cycles.values(),default=0),1)
            cycles+=duration
            expected=sum(macs(w,tile) for tile,_ in tiles)
            if opmac!=expected:raise ValueError('CNN logical MAC coverage mismatch')
            totals['logical_macs']+=opmac;totals['padded_macs']+=oppadded;totals['vector_ops']+=opvec
            rows.append({'operator':op.id,'kind':op.kind,'cycles':duration,'node_service_cycles':node_service,
                         'dram_service_cycles':external_cycles,'logical_macs':opmac,'padded_macs':oppadded,
                         'vector_ops':opvec,'payload_service_bytes':opbytes,'tile_count':len(tiles),
                         'placements':{f'c{node}':n for node,n in bynode.items()}})
        # Return user-visible outputs to the external allocation even when an
        # action kept the producing layer in GB. Alias roots share ownership.
        external_cycles=0;link_cycles=defaultdict(int);opbytes=0
        for root in {self.roots[name] for name in w.outputs}:
            for region,source in layouts[root][0]:
                if source=='dram':continue
                width=region.bounds[-1][1]-region.bounds[-1][0]
                nr=region.elements//width
                transfer('dram',source,region.elements,nr*ceildiv(width+15,64),nr*ceildiv(width+15,16),'output',True)
        export_cycles=max(external_cycles,max(link_cycles.values(),default=0))
        if export_cycles:
            cycles+=export_cycles
            rows.append({'operator':'host_output_export','kind':'DMA','cycles':export_cycles,'logical_macs':0,'tile_count':0,'placements':{}})
        cycles=max(1,cycles)
        traffic['hierarchy_movement_bytes']=dict(traffic['hierarchy_movement_bytes'])
        return {'model':'race_gemmini_cnn_hierarchy_v1','chiplets':count,'mesh_shape':mesh_shape(count),
                'memory_hierarchy':h.record(),'platform_scenario':asdict(p),'source_provenance':c.provenance(),
                'modeled_cycles':cycles,'operations':totals,'traffic':traffic,'layers':rows,
                'storage':{'gb_activation_peak_bytes_by_node':peak,'gb_activation_capacity_bytes':self.activation_cache_bytes,
                           'external_static_allocation_bytes':dram_allocation},
                'operator_placement':placements,'directed_links':[{'from':a,'to':b,'bytes':amount,'flows':link_flows[a,b]} for (a,b),amount in sorted(links.items())],
                'dependency_link_bytes':[{'producer':source,'consumer':destination,'from':a,'to':b,'direction':direction,'bytes':amount}
                     for (source,destination,a,b,direction),amount in sorted(attribution.items())],
                'largest_transfers':[item[2] for item in sorted(largest,reverse=True)],'transfer_count':flow_serial,
                'identity':digest({'design':design,'workload':w,'memory':h.record(),'platform':p,'model':'cnn_hierarchy_v1'}),
                'timing_status':'LAYER_BARRIER_BOTTLENECK_ESTIMATE_UNCALIBRATED',
                'coverage':{'whole_cnn_operator_graph':True,'precision':'INT8 activations/weights; INT32 MAC; FP32 vector postprocess',
                            'grouped_depthwise_dilated_conv':True,'activation_residency':True,'peer_activation_mesh':True,
                            'h_w_output_channel_partition':True,'input_channel_k_blocking':True,
                            'cross_chiplet_k_reduction':False,'interlayer_pipeline':False,'quantization_accuracy_proven':False},
                'assumptions':['GB bank 0 is 64 KiB live activation cache; bank 1 streams gather/A/output; banks 2/3 hold B.',
                               'Gather reads row spans separately per kernel tap; stride gaps are charged; transaction alignment uses conservative 16-byte offset bound.',
                               'Streamed im2col packing and all non-GEMM operators use a project-added 16-lane FP32 vector unit.',
                               'Vector throughput, port rates, and layer timing are research assumptions; no cycle-accurate Gemmini or silicon claim.',
                               'Layer time is maximum declared resource service; no interlayer overlap. Not a realizable-schedule proof or latency upper bound.',
                               'GB capacity failures reject the whole selected action; no silent spill/retile/legacy-model fallback.']}
