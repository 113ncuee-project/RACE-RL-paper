"""Learnable relational message passing and hierarchical masked action probabilities."""
import torch
from torch import nn
from .observation import TILE_FEATURE_SIZE
from .requirements import TASK_FEATURE_SIZE


class RelationalEncoder(nn.Module):
    def __init__(self,width=32,variant="full",schema_version=5):
        super().__init__()
        if variant not in {"full","compact","dnngraph","hardwaregraph","no_attribution","no_criticality"}:
            raise ValueError("unknown state ablation")
        if schema_version not in (5,6,7,8): raise ValueError('unsupported encoder schema')
        self.variant,self.width,self.schema_version = variant,width,schema_version
        # Schemas 6/7 add exact output-block coordinates and Gemmini execution state.
        extra=25 if schema_version==8 else 6 if schema_version in (6,7) else 0
        self.tile = nn.Sequential(nn.Linear(TILE_FEATURE_SIZE+extra,width),nn.Tanh())
        self.hardware = nn.Sequential(nn.Linear(9,width),nn.Tanh())
        self.dnn_message = nn.Linear(width,width,bias=False)
        self.hw_message = nn.Linear(width,width,bias=False)
        self.memory_message = nn.Linear(width,width,bias=False)
        self.cross_message = nn.Linear(width,width,bias=False)
        self.link_message = nn.Linear(4,width,bias=False)
        self.global_net = nn.Sequential(nn.Linear((22 if schema_version in (6,7,8) else 10)+TASK_FEATURE_SIZE+2*width,width),nn.Tanh())

    def forward(self,o):
        if o.schema_version != self.schema_version:
            raise ValueError("incompatible observation schema; old checkpoints require their frozen source")
        dtype = self.tile[0].weight.dtype
        tf = torch.tensor(o.tile_features,dtype=dtype)
        hf = torch.tensor(o.hardware_features,dtype=dtype)
        ef = torch.tensor(o.hardware_edge_features,dtype=dtype).reshape(-1,4)
        if self.variant == "no_criticality":
            tf[:,TILE_FEATURE_SIZE-2:TILE_FEATURE_SIZE] = 0
            ef[:,-1] = 0
        t,h = self.tile(tf),self.hardware(hf)
        tm,hm = torch.zeros_like(t),torch.zeros_like(h)
        if self.variant in {"full","dnngraph","no_attribution","no_criticality"}:
            for a,b in o.dependency_edge_index:
                tm[b] = tm[b]+self.dnn_message(t[a])
                tm[a] = tm[a]+self.dnn_message(t[b])
        if self.variant in {"full","hardwaregraph","no_attribution","no_criticality"}:
            for a,b in o.hardware_edge_index:
                hm[b] = hm[b]+self.hw_message(h[a])
            for a,b in o.memory_port_edge_index:
                hm[b] = hm[b]+self.memory_message(h[a])
        if self.variant in {"full","no_attribution","no_criticality"}:
            for a,b in o.placement_edge_index:
                tm[a] = tm[a]+self.cross_message(h[b])
                hm[b] = hm[b]+self.cross_message(t[a])
        if self.variant in {"full","no_criticality"}:
            seen = set()
            for d,l,flow,_ in o.dependency_link_relations:
                # Shared flow-link attributed once per dependency, without duplicating energy ledger.
                if (d,l,flow) in seen: continue
                seen.add((d,l,flow))
                a,b = o.dependency_edge_index[d]
                message = self.link_message(ef[l])
                tm[a],tm[b] = tm[a]+message,tm[b]+message
        t,h = torch.tanh(t+tm),torch.tanh(h+hm)
        if self.variant == "compact":
            t = t.mean(0).expand_as(t)
            h = h.mean(0).expand_as(h)
        glob = torch.tensor(o.global_features,dtype=dtype)
        # Raw SI power/latency are transformed without fitted/test-set statistics.
        glob[4:6] = torch.log(torch.clamp(glob[4:6],min=1e-30))/30 if glob[8] else 0
        g = self.global_net(torch.cat((glob,torch.tensor(o.task_features,dtype=dtype),t.mean(0),h.mean(0))))
        return t,h,g


class AutoregressivePolicy(nn.Module):
    def __init__(self,width=32,state_variant="full"):
        super().__init__()
        self.encoder = RelationalEncoder(width,state_variant)
        self.root_score = nn.Linear(2*width,1)
        self.scope_score = nn.Linear(3*width,1)
        self.type_embeddings = nn.Embedding(3,width)
        self.type_score = nn.Linear(4*width,1)
        self.parameter_score = nn.Sequential(nn.Linear(6*width+1,width),nn.Tanh(),nn.Linear(width,1))
        self.design_family = nn.Linear(width,3)
        self.count_embedding = nn.Sequential(nn.Linear(4,width),nn.Tanh())
        self.count_score = nn.Linear(2*width,1)
        self.retile_score = nn.Sequential(nn.Linear(3*width+2,width),nn.Tanh(),nn.Linear(width,1))

    def distribution(self,o,_encoded=None):
        from .architecture import DesignObservation
        if isinstance(o,DesignObservation):
            return self.design_distribution(o)
        if not o.legal_actions:
            raise ValueError("no legal action distribution")
        t,h,g = self.encoder(o) if _encoded is None else _encoded
        ti = {name:i for i,name in enumerate(o.tile_ids)}
        hi = {name:i for i,name in enumerate(o.hardware_ids)}
        positions = [i for i,mask in enumerate(o.legal_action_masks) if mask]
        actions = [o.legal_actions[i] for i in positions]
        if not actions: raise ValueError("all actions masked")
        # Group once. Parameter heads evaluate one batch rather than one
        # tiny tensor operation for every candidate.
        from collections import defaultdict
        groups = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for i,a in enumerate(actions): groups[a.root][a.scope][a.kind].append(i)
        roots = sorted(groups)
        root_inputs = torch.cat((g.expand(len(roots),-1),t[[ti[r] for r in roots]]),1)
        root_logs = dict(zip(roots,torch.log_softmax(self.root_score(root_inputs).squeeze(-1),0)))
        type_ids = {"cut_migrate":0,"work_share":1,"memory_affinity":2}
        scopes = sorted({a.scope for a in actions})
        scope_vectors = {scope:t[[ti[name] for name in scope]].mean(0) for scope in scopes}
        scope_matrix = torch.stack([scope_vectors[scope] for scope in scopes])
        scope_index = {scope:i for i,scope in enumerate(scopes)}
        contexts = []
        for root in roots:
            root_scopes = sorted(groups[root])
            scope_inputs = torch.cat((g.expand(len(root_scopes),-1),t[ti[root]].expand(len(root_scopes),-1),
                torch.stack([scope_vectors[scope] for scope in root_scopes])),1)
            scope_logs = torch.log_softmax(self.scope_score(scope_inputs).squeeze(-1),0)
            for scope,scope_log in zip(root_scopes,scope_logs):
                kinds = sorted(groups[root][scope])
                type_inputs = torch.cat((g.expand(len(kinds),-1),t[ti[root]].expand(len(kinds),-1),
                    scope_vectors[scope].expand(len(kinds),-1),
                    self.type_embeddings.weight[[type_ids[k] for k in kinds]]),1)
                type_logs = torch.log_softmax(self.type_score(type_inputs).squeeze(-1),0)
                for kind,type_log in zip(kinds,type_logs):
                    contexts.append((groups[root][scope][kind],root_logs[root]+scope_log+type_log))
        compute_vectors = h.new_zeros((len(actions),h.shape[1]))
        with_compute = [i for i,a in enumerate(actions) if a.compute_target]
        if with_compute:
            compute_vectors[with_compute] = h[[hi[actions[i].compute_target] for i in with_compute]]
        param_inputs = torch.cat((g.expand(len(actions),-1),t[[ti[a.root] for a in actions]],
            scope_matrix[[scope_index[a.scope] for a in actions]],
            self.type_embeddings.weight[[type_ids[a.kind] for a in actions]],
            h[[hi[a.target] for a in actions]],compute_vectors,
            g.new_tensor([[a.count/4] for a in actions])),1)
        param_logits = self.parameter_score(param_inputs).squeeze(-1)
        leaves = g.new_empty(len(actions))
        for indices,base in contexts:
            leaves[indices] = base+torch.log_softmax(param_logits[indices],0)
        joint = g.new_full((len(o.legal_actions),),float('-inf'))
        joint[positions] = leaves
        return torch.distributions.Categorical(logits=joint),self.value(g)

    def design_distribution(self,o):
        from .architecture import CountAction,RetileAction,IndependentAction
        t,h,g = self.encoder(o.graph)
        families = [[],[],[]]
        for i,a in enumerate(o.legal_actions):
            families[1 if isinstance(a,CountAction) else 2 if isinstance(a,(RetileAction,IndependentAction)) else 0].append(i)
        available = [k for k,indices in enumerate(families) if any(o.legal_action_masks[i] for i in indices)]
        if not available: raise ValueError('all design actions masked')
        def mixed(logits):
            return 0.95*torch.softmax(logits,0)+0.05/len(logits)
        gate = mixed(self.design_family(g)[available])
        log_gate = dict(zip(available,torch.log(gate)))
        logs = g.new_full((len(o.legal_actions),),float('-inf'))
        ti = {name:i for i,name in enumerate(o.graph.tile_ids)}
        hi = {name:i for i,name in enumerate(o.graph.hardware_ids)}
        for family in available:
            local = [j for j,i in enumerate(families[family]) if o.legal_action_masks[i]]
            indices = [families[family][j] for j in local]
            if family == 0:
                from dataclasses import replace
                masks = tuple(o.legal_action_masks[i] for i in families[0])
                distribution,_ = self.distribution(replace(o.graph,legal_action_masks=masks),_encoded=(t,h,g))
                conditional = 0.95*distribution.probs[local]+0.05/len(local)
            elif family == 1:
                features = g.new_tensor([o.count_features[j] for j in local])
                embedded = self.count_embedding(features)
                logits = self.count_score(torch.cat((g.expand(len(local),-1),embedded),1)).squeeze(-1)
                conditional = mixed(logits)
            else:
                features = [o.retile_features[j] for j in local]
                groups = list(dict.fromkeys(ids for ids,_,_,_ in features))
                group_index = {ids:j for j,ids in enumerate(groups)}
                pooled = torch.stack([t[[ti[name] for name in ids]].mean(0) for ids in groups])
                vectors = torch.cat((g.expand(len(local),-1),pooled[[group_index[ids] for ids,_,_,_ in features]],
                    h[[hi[target] for _,target,_,_ in features]],
                    g.new_tensor([[count,spread] for _,_,count,spread in features])),1)
                conditional = mixed(self.retile_score(vectors).squeeze(-1))
            logs[indices] = log_gate[family]+torch.log(conditional)
        return torch.distributions.Categorical(logits=logs),self.value(g)

    def value(self,g):
        # Formal policy has no value predictor. Comparison models may override.
        return None

    @torch.no_grad()
    def select(self,o):
        distribution,value = self.distribution(o)
        index = distribution.sample()
        return o.legal_actions[index.item()],distribution.log_prob(index).item(),value.item() if value is not None else None
