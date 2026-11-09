"""One schema-7 relational encoder and actor for complete Gemmini transactions."""
import torch
from torch import nn
from ..policy import RelationalEncoder, AutoregressivePolicy


class GemminiPolicy(nn.Module):
    def __init__(self, width=32, state_variant="full"):
        super().__init__()
        self.encoder = RelationalEncoder(width, state_variant, schema_version=7)
        self.actor = nn.Sequential(nn.Linear(3*width+12, width), nn.Tanh(), nn.Linear(width, 1))

    def distribution(self, observation):
        if observation.backend != "race_gemmini_hierarchy_ws_v2": raise ValueError("wrong observation backend")
        t, h, g = self.encoder(observation.graph)
        allowed = [index for index, flag in enumerate(observation.legal_action_masks) if flag]
        if not allowed: raise ValueError("all Gemmini actions masked")
        hw_index = {name:index for index, name in enumerate(observation.graph.hardware_ids)}
        targets = h[[hw_index[f"c{observation.legal_actions[index].target % observation.compute_count}.array"]
                     for index in allowed]]
        # A target beyond the current count is represented by its modulo anchor;
        # the proposed count/target remain explicit in candidate features.
        features = g.new_tensor([observation.action_features[index] for index in allowed])
        context = torch.cat((g.expand(len(allowed), -1), t.mean(0).expand(len(allowed), -1), targets, features), 1)
        logits = self.actor(context).squeeze(-1)
        probabilities = 0.95*torch.softmax(logits, 0)+0.05/len(allowed)
        all_probabilities = g.new_zeros(len(observation.legal_actions))
        all_probabilities[allowed] = probabilities
        return torch.distributions.Categorical(probs=all_probabilities), None

    select = AutoregressivePolicy.select
