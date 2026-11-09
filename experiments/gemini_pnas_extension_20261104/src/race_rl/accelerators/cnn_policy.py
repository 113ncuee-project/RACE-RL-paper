"""A single shared actor samples a complete joint transaction in one forward pass."""
import torch
from torch import nn
from ..policy import RelationalEncoder
from .cnn_design import transaction,PROFILES,coupled_scope


class CNNPolicy(nn.Module):
    structured_actions=True
    def __init__(self,width=32,state_variant='full'):
        super().__init__()
        self.encoder=RelationalEncoder(width,state_variant,schema_version=8)
        self.actor=nn.ModuleDict({'count':nn.Linear(width,16),'anchor':nn.Linear(2*width,1),
            'profile':nn.Linear(2*width,len(PROFILES)),'residency':nn.Linear(2*width,4),
            'layout':nn.Linear(2*width,2),'target':nn.Linear(2*width,16)})

    @staticmethod
    def categorical(logits):
        probs=.95*torch.softmax(logits,0)+.05/logits.numel()
        return torch.distributions.Categorical(probs=probs)

    def _decisions(self,observation,choices=None):
        if observation.backend!='race_gemmini_cnn_hierarchy_v1':raise ValueError('wrong CNN observation backend')
        t,h,g=self.encoder(observation.graph)
        chosen=[];logs=[];entropies=[]
        def choose(distribution,index,mapping=None):
            value=distribution.sample() if choices is None else torch.tensor(mapping.index(choices[index]) if mapping else choices[index])
            chosen.append(mapping[int(value)] if mapping else int(value));logs.append(distribution.log_prob(value));entropies.append(distribution.entropy())
        choose(self.categorical(self.actor['count'](g)),0)
        anchors=observation.anchor_indices
        logits=self.actor['anchor'](torch.cat((g.expand(len(anchors),-1),t[list(anchors)]),1)).squeeze(-1)
        choose(self.categorical(logits),1,anchors)
        scope=coupled_scope(observation.workload,chosen[1],observation.max_scope)
        for offset,operator in enumerate(scope):
            context=torch.cat((g,t[operator]))
            for index,name in ((2,'profile'),(3,'residency'),(4,'layout')):
                choose(self.categorical(self.actor[name](context)),index+4*offset)
            choose(self.categorical(self.actor['target'](context)[:chosen[0]+1]),5+4*offset)
        return tuple(chosen),torch.stack(logs).sum(),torch.stack(entropies).sum()

    def select(self,observation):
        with torch.no_grad():
            choices,logp,_=self._decisions(observation)
            action=transaction(observation.workload,observation.current_design,choices,observation.max_scope)
        return action,float(logp),None

    def evaluate_action(self,observation,action):
        if action!=transaction(observation.workload,observation.current_design,action.choices,observation.max_scope):
            raise ValueError('trajectory action does not belong to observation')
        _,logp,entropy=self._decisions(observation,action.choices)
        return logp,entropy
