from inspect import signature
import torch
from torch import nn
from torch_geometric.nn import (
    SAGEConv,
    APPNP,
    GCNConv,
    global_max_pool,
    global_add_pool,
    global_mean_pool,
    BatchNorm,
    GraphSizeNorm,
    InstanceNorm,
)
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import softmax

from a2c_ppo_acktr.model import NNBase, Policy
from a2c_ppo_acktr.utils import init

from .distributions import NodeObjective
from .functions import *


class GraphPolicy(Policy):
    """
    Wraps Policy class to handle graph inputs via receipts
    """

    def __init__(self, *args, receipts, **kwargs):
        super(GraphPolicy, self).__init__(*args, **kwargs)
        self.receipts = receipts
        # patch distributions to handle node based selection
        self.dist = NodeObjective()

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        inputs = self.receipts.redeem(inputs)
        return super(GraphPolicy, self).act(inputs, rnn_hxs, masks, deterministic)

    def get_value(self, inputs, rnn_hxs, masks):
        inputs = self.receipts.redeem(inputs)
        return super(GraphPolicy, self).get_value(inputs, rnn_hxs, masks)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        inputs = self.receipts.redeem(inputs)
        return super(GraphPolicy, self).evaluate_actions(inputs, rnn_hxs, masks, action)


class EdgeAttrs(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(EdgeAttrs, self).__init__()
        self.edge_fn = nn.Sequential(
            init_ot(nn.Linear(input_dim * 3, input_dim), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(input_dim, out_dim), "sigmoid"),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index):
        x_src = x[edge_index[0]]
        x_dst = x[edge_index[1]]
        _x = torch.cat([x_src, x_dst, x_src - x_dst], dim=1)
        return self.edge_fn(_x)


class EdgeMask(nn.Module):
    def __init__(self, edge_dim, mask_dim=1, K=5, bias=True):
        super(EdgeMask, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1e-3))
        self.edge_fn = nn.Sequential(
            init_xn(nn.Linear(edge_dim, 1), "sigmoid"), nn.Sigmoid()
        )
        self.conv = APPNP(K=K, alpha=self.alpha)
        self.K = K
        if bias:
            self.bias = nn.Parameter(torch.ones(mask_dim) * -3)
        else:
            sekf.bias = None

    def forward(self, edge_attr, mask, edge_index):
        edge_weights = self.edge_fn(edge_attr).view(-1)
        # assert False, str((edge_attr.shape, mask.shape, edge_index.shape))
        fill = torch.relu(mask)
        fill = self.conv(fill, edge_index, edge_weights)
        if self.bias is not None:
            fill = fill - F.softplus(self.bias)
        fill = torch.tanh(fill)
        return fill, edge_weights


class DirectionalPropagation(nn.Module):
    def __init__(self, hidden_size, transitivity_size, edge_attr_size, mask_dim, K):
        super(DirectionalPropagation, self).__init__()
        edge_dim = transitivity_size + edge_attr_size
        self.dom_transitivity_fn = EdgeAttrs(hidden_size, transitivity_size)
        self.dom_edge_mask = EdgeMask(edge_dim, mask_dim, K)
        self.pos_edge_mask = EdgeMask(edge_dim, mask_dim, K)

    def forward(self, x, dom, projection, mask):
        spatial_edge_trans = self.dom_transitivity_fn(x, dom.spatial_edge_index)
        dom_edge_trans = self.dom_transitivity_fn(x, dom.dom_edge_index)
        _spatial_edge_trans = safe_bc_edges(
            spatial_edge_trans, projection.br_spatial_edge_index
        )
        pos_mask, pos_mask_ew = self.pos_edge_mask(
            torch.cat([projection.spatial_edge_attr, _spatial_edge_trans], dim=1),
            mask,
            projection.spatial_edge_index,
        )
        _dom_edge_trans = safe_bc_edges(dom_edge_trans, projection.br_dom_edge_index)
        dom_mask, dom_mask_ew = self.dom_edge_mask(
            torch.cat([projection.dom_edge_attr, _dom_edge_trans], dim=1),
            mask,
            projection.dom_edge_index,
        )
        return torch.max(mask, torch.max(pos_mask, dom_mask))


class FixedActionDecoder(nn.Module):
    def __init__(self):
        super(FixedActionDecoder, self).__init__()
        self.default_action = 1  # input goal
        self.action_words = [
            ["click", "submit", "target", "focus"],
            ["select", "text", "noun", "username", "password"],
            ["copy"],
            ["paste"],
        ]
        from .domx import short_embed

        self.action_index = [i for i, a in enumerate(self.action_words) for word in a]
        self.action_vectors = torch.cat(
            [
                torch.as_tensor(short_embed(word)).reshape(1, -1, 1)
                for i, a in enumerate(self.action_words)
                for word in a
            ],
            dim=2,
        )

    def forward(self, embedded_words):
        batch_size = embedded_words.shape[0]
        point_size = len(self.action_index)
        action_size = len(self.action_words)
        device = embedded_words.device
        sims = (
            F.cosine_similarity(
                embedded_words.reshape(batch_size, -1, 1).repeat(1, 1, point_size),
                self.action_vectors.repeat(batch_size, 1, 1).to(device),
                dim=1,
            )
            .view(batch_size, point_size)
            .view(-1)
        )
        batch_action_index = (
            torch.tensor(self.action_index, dtype=torch.long).repeat(batch_size)
            + action_size
            * torch.arange(0, batch_size)
            .view(batch_size, 1)
            .repeat(1, point_size)
            .view(-1)
        ).to(device)
        batchwise_action_sims = global_max_pool(sims, batch_action_index).view(
            batch_size, action_size
        )

        action_idx = (
            batchwise_action_sims.argmax(dim=1)
            + torch.arange(0, batch_size, device=device) * action_size
        ).view(batch_size)
        ret = torch.zeros(batch_size * action_size, device=device)
        ret[action_idx] = 1.0
        return ret.view(batch_size, action_size)


def submodule(target_domain=None):
    def f(func):
        func.__target_domain__ = target_domain
        return func

    return f


class GNNBase(NNBase):
    def __init__(
        self,
        input_dim,
        dom_encoder=None,
        recurrent=False,
        hidden_size=128,
        text_embed_size=25,
    ):
        super(GNNBase, self).__init__(recurrent, hidden_size, hidden_size)
        self.hidden_size = hidden_size
        self.text_embed_size = text_embed_size

        self.dom_encoder = dom_encoder

        edge_attr_size = 4
        action_one_hot_size = 4  # 5 to enable wait
        query_input_dim = 9 + 2 * text_embed_size
        if dom_encoder:
            query_input_dim += dom_encoder.out_channels
        self.attr_norm = nn.BatchNorm1d(text_embed_size)
        self.action_decoder = FixedActionDecoder()
        self.global_conv = SAGEConv(query_input_dim, hidden_size - query_input_dim)
        self.global_dropout = nn.Dropout(p=0.3)

        self.dom_ux_action_fn = nn.Sequential(
            # paste  & copy have equal weight
            init_xn(nn.Linear(hidden_size, action_one_hot_size - 1), "linear"),
            nn.Softmax(dim=1),
        )
        transitivity_size = 4
        self.leaf_prop = DirectionalPropagation(
            hidden_size, transitivity_size, edge_attr_size, mask_dim=1, K=5
        )
        self.goal_prop = DirectionalPropagation(
            hidden_size, transitivity_size, edge_attr_size, mask_dim=1, K=5
        )
        self.value_prop = DirectionalPropagation(
            hidden_size, transitivity_size, edge_attr_size, mask_dim=1, K=5
        )

        self.dom_description_fn = nn.Sequential(
            init_xu(nn.Linear(hidden_size, text_embed_size), "tanh"), nn.Tanh()
        )
        # attr similarities
        self.attr_similarity_size = attr_similarity_size = 10
        # [ goal att, value ]
        self.dom_objective_attr_fn = nn.Sequential(
            init_xn(nn.Linear(hidden_size, 2 * attr_similarity_size), "linear"),
            nn.Softplus(),
        )
        # [ enabled ]
        self.dom_objective_enable_size = 3
        self.dom_objective_enable_fn = nn.Sequential(
            init_xn(nn.Linear(hidden_size, self.dom_objective_enable_size), "tanh"),
            nn.Tanh(),
        )

        self.objective_ux_action_fn = nn.Sequential(
            # init_r(nn.Linear(text_embed_size, text_embed_size)),
            # nn.ReLU(),
            self.action_decoder,
            # init_t(nn.Linear(action_one_hot_size, action_one_hot_size)),
            # nn.Softmax(dim=1),
            # nn.Linear(action_one_hot_size, action_one_hot_size),
            # nn.Softmax(dim=1),
        )
        # enabled, value similarity, ac, ux action
        self.objective_indicator_size = objective_indicator_size = (
            2 + self.dom_objective_enable_size + action_one_hot_size
        )
        self.status_indicators_fn = nn.Sequential(
            init_xn(
                nn.Linear(objective_indicator_size, objective_indicator_size), "tanh"
            ),
            nn.Tanh(),
        )
        # order, is last, has focus
        objective_active_size = objective_indicator_size + 3
        if recurrent:
            # dom + pos + goal indicators
            goal_dom_size = hidden_size + objective_active_size
            self.goal_dom_encoder = GCNConv(goal_dom_size, hidden_size)
            self.state_indicator = nn.Sequential(
                init_xn(nn.Linear(goal_dom_size + hidden_size, hidden_size), "relu"),
                nn.ReLU(),
                init_xu(nn.Linear(hidden_size, 1), "tanh"),
                nn.Tanh(),
            )
            objective_active_size += 1
        self.objective_active = nn.GRU(
            input_size=objective_active_size,
            hidden_size=objective_active_size,
            num_layers=2,
            # bidirectional=True,
        )
        # consensus, dom interest, leaf att, goal mask, ux action mask, [Goal indicators], [GRU indicators]
        trunk_size = 5 + objective_active_size + objective_indicator_size
        # norm consensus
        self.trunk_norm = InstanceNorm(1)
        self.actor_gate = nn.Sequential(
            init_xu(nn.Linear(trunk_size, trunk_size), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(trunk_size, 1), "relu"),
            nn.ReLU(),
        )

        critic_embed_size = 16
        critic_size = trunk_size
        self.critic_mp_score = nn.Sequential(
            init_xu(nn.Linear(critic_size, critic_embed_size), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(critic_embed_size, critic_embed_size), "sigmoid"),
            nn.Sigmoid(),
        )
        self.critic_ap_score = nn.Sequential(
            init_xu(nn.Linear(critic_size, critic_embed_size), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(critic_embed_size, critic_embed_size), "sigmoid"),
            nn.Sigmoid(),
        )
        critic_gate_size = (
            2 * critic_embed_size + 2 * objective_active_size
        )  # active steps, near_completion
        self.critic_gate = nn.Sequential(
            init_xu(nn.Linear(critic_gate_size, critic_embed_size), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(critic_embed_size, 1), "linear"),
        )

    def resolve(self, func, resolving=None):
        """
        Call a method whose arguments are tensors resolved by matching their name to a self method
        """
        fname = func.__name__
        deps = signature(func)

        if fname not in self.last_tensors:
            # collect dependencies
            resolving = resolving or {fname}
            kwargs = {}
            for param in deps.parameters:
                if param == "self":
                    continue
                if param not in self.last_tensors:
                    assert (
                        param not in resolving
                    ), f"{fname} has circular dependency through {param}"
                    resolving.add(param)
                    assert hasattr(self, param), f"{fname} has unmet dependency {param}"
                    self.last_tensors[param] = self.resolve(
                        getattr(self, param), resolving
                    )
                kwargs[param] = self.last_tensors[param]
            result = func(**kwargs)
            assert result is not None, f"Bad return value: {fname}"
            self.last_tensors[fname] = result
        return self.last_tensors[fname]

    @submodule("dom")
    def x(self, dom):
        return torch.cat(
            [
                dom.tag,
                dom.classes,
                dom.radio_value,
                dom.rx,
                dom.ry,
                dom.width,
                dom.height,
                dom.top,
                dom.left,
                dom.focused,
                dom.depth,
            ],
            dim=1,
        ).clamp(-1, 1)

    @submodule("dom")
    def full_x(self, dom, x):
        if self.dom_encoder:
            _add_x = torch.tanh(
                self.dom_encoder(
                    torch.cat(
                        [
                            dom.text,
                            dom.value,
                            dom.tag,
                            dom.classes,
                            dom.rx,
                            dom.ry,
                            dom.width,
                            dom.height,
                            dom.top,
                            dom.left,
                        ],
                        dim=1,
                    ),
                    dom.dom_edge_index,
                )
            )
            x = torch.cat([x, _add_x], dim=1)

        _add_x = self.global_conv(x, dom.spatial_edge_index)

        x = torch.cat([x, _add_x], dim=1)
        self.last_tensors["conv_x"] = _add_x
        full_x = self.global_dropout(x)
        return full_x

    @submodule("dom_leaf")
    def leaves_att(self, dom, dom_leaf, full_x):
        leaf_t_mask = dom_leaf.mask.type(torch.float)
        return self.leaf_prop(full_x, dom, dom_leaf, leaf_t_mask)

    @submodule("dom")
    def dom_ux_action(self, full_x):
        # infer action from dom nodes
        dom_ux_action = self.dom_ux_action_fn(full_x)
        # for dom action input == paste
        return torch.cat([dom_ux_action[:, 0:], dom_ux_action[:, 1].view(-1, 1)], dim=1)

    @submodule("dom_leaf")
    def leaf_ux_action(self, dom_ux_action, dom_leaf):
        return global_max_pool(
            safe_bc(dom_ux_action, dom_leaf.dom_index) * dom_leaf.mask,
            dom_leaf.leaf_index,
        )

    @submodule("dom_field")
    def obj_tag_similarities(self, dom, field, dom_field, full_x):
        # compute an objective dependent dom feats
        # start with word embedding similarities
        obj_sim = lambda tag, obj: F.cosine_similarity(
            safe_bc(dom[tag], dom_field.dom_index),
            safe_bc(field[obj], dom_field.field_index),
            dim=1,
        ).view(-1, 1)
        dom_desc = self.dom_description_fn(full_x)
        _dom_desc = safe_bc(dom_desc, dom_field.dom_index)
        dom_desc_key_sim = F.cosine_similarity(
            _dom_desc, safe_bc(field.key, dom_field.field_index), dim=1
        ).view(-1, 1)
        dom_desc_query_sim = F.cosine_similarity(
            _dom_desc, safe_bc(field.query, dom_field.field_index), dim=1
        ).view(-1, 1)
        obj_tag_similarities = torch.relu(
            torch.cat(
                [
                    obj_sim("text", "key"),
                    obj_sim("text", "query"),
                    obj_sim("value", "key"),
                    obj_sim("value", "query"),
                    obj_sim("classes", "key"),
                    obj_sim("classes", "query"),
                    obj_sim("tag", "key"),
                    obj_sim("tag", "query"),
                    dom_desc_key_sim,
                    dom_desc_query_sim,
                ],
                dim=1,
            )
        )
        if SAFETY:
            assert (
                global_max_pool(obj_tag_similarities.max(dim=1)[0], dom_field.batch)
                .min()
                .item()
                > 0
            )
        return obj_tag_similarities

    @submodule("dom_field")
    def dom_obj_attr(self, dom_field, full_x):
        # [:,:,0] = goal target. [:,:,2] = goal completion. ; odds are cuttoff
        dom_obj_attr = self.dom_objective_attr_fn(full_x).view(
            full_x.shape[0], self.attr_similarity_size, 2
        )
        return safe_bc(dom_obj_attr, dom_field.dom_index)

    @submodule("dom_field")
    def obj_att_input(self, dom, dom_field, full_x, dom_obj_attr, obj_tag_similarities):
        # infer interested dom attrs
        dom_obj_int = dom_obj_attr[:, :, 0]
        # ux action tag sensitivities
        obj_att_input = torch.relu(
            (dom_obj_int * obj_tag_similarities).max(dim=1, keepdim=True).values
        )

        if SAFETY:
            assert global_max_pool(obj_att_input, dom_field.batch).min().item() > 0

        return self.goal_prop(full_x, dom, dom_field, obj_att_input)

    @submodule("field")
    def obj_ux_action(self, field):
        return self.objective_ux_action_fn(field.key)

    @submodule("action")
    def ux_action_consensus(self, action, obj_ux_action, leaf_ux_action):
        # project into main action space
        _obj_ux_action = safe_bc(
            obj_ux_action.view(-1, 1), action.field_ux_action_index
        )

        _leaf_ux_action = safe_bc(
            leaf_ux_action.view(-1, 1), action.leaf_ux_action_index
        )
        return _leaf_ux_action * _obj_ux_action

    @submodule("action")
    def dom_interest(self, action, obj_att_input, leaves_att):
        # project into main action space
        _leaf_mask = safe_bc(leaves_att, action.leaf_dom_index)

        if SAFETY:
            assert global_max_pool(_leaf_mask, action.batch).min().item() > 0

        _obj_att_input = safe_bc(obj_att_input, action.field_dom_index)
        return _leaf_mask * _obj_att_input

    @submodule("action")
    def action_consensus(self, action, dom_interest, ux_action_consensus):
        action_consensus = dom_interest * ux_action_consensus
        if SAFETY:
            assert global_max_pool(action_consensus, action.batch).min().item() > 0
        return action_consensus

    @submodule("action")
    def ac_norm(self, action, action_consensus):
        # norm activity within each goal
        _ac = global_max_pool(action_consensus, action.action_index)
        return self.trunk_norm(
            _ac, global_max_pool(action.field_index, action.action_index)
        )

    @submodule("dom_field")
    def value_mask(self, dom, dom_field, full_x, dom_obj_attr, obj_tag_similarities):
        # compute objective completeness with goal word similarities
        # [0 - 1]
        dom_obj_comp_attr = dom_obj_attr[:, :, 1]
        value_mask = (
            (dom_obj_comp_attr * obj_tag_similarities).max(dim=1, keepdim=True).values
        )
        return self.value_prop(full_x, dom, dom_field, value_mask)

    @submodule("dom")
    def enabled_mask(self, full_x):
        return self.dom_objective_enable_fn(full_x)

    @submodule("action")
    def action_status_indicators(
        self, dom, action, action_consensus, enabled_mask, value_mask, obj_ux_action
    ):
        action_status_indicators = torch.cat(
            [
                safe_bc(value_mask, action.field_dom_index),
                safe_bc(enabled_mask, action.dom_index),
                self.trunk_norm(action_consensus, action.field_index),
                safe_bc(obj_ux_action, action.field_index),
            ],
            dim=1,
        )
        return self.status_indicators_fn(action_status_indicators)

    @submodule("field")
    def obj_indicator_order(
        self,
        dom,
        field,
        dom_field,
        action,
        action_consensus,
        action_status_indicators,
        full_x,
        rnn_hxs,
        masks,
    ):
        # begin determining active step
        status_indicators = global_max_pool(
            action_status_indicators, action.field_index
        )
        dom_obj_focus = safe_bc(dom.focused, action.dom_index)

        # pack on order embed and goal focus
        obj_indicator_order = torch.cat(
            [
                status_indicators,
                field.order,
                field.is_last,
                global_max_pool(dom_obj_focus * action_consensus, action.field_index),
            ],
            dim=1,
        )

        if self.is_recurrent:
            goal_dom_input = torch.cat(
                [
                    safe_bc(full_x, dom_field.dom_index),
                    safe_bc(obj_indicator_order, dom_field.field_index),
                ],
                dim=1,
            )
            goal_dom_encoded = self.goal_dom_encoder(
                goal_dom_input, dom_field.dom_edge_index
            )
            goal_dom_flat = global_max_pool(goal_dom_encoded, dom_field.batch)
            curr_state, rnn_hxs = self._forward_gru(goal_dom_flat, rnn_hxs, masks)
            self.last_tensors["rnn_hxs"] = rnn_hxs
            state_indicator_input = torch.cat(
                [goal_dom_input, safe_bc(curr_state, dom_field.batch)], dim=1
            )
            state_indicator = global_max_pool(
                self.state_indicator(state_indicator_input), dom_field.field_index
            )
            obj_indicator_order = torch.cat(
                [state_indicator, obj_indicator_order], dim=1
            )
        return obj_indicator_order

    @submodule("field")
    def obj_active(self, field, obj_indicator_order):
        obj_ind_seq = pack_as_sequence(obj_indicator_order, field.batch)
        obj_active_seq, obj_active_mem = self.objective_active(obj_ind_seq)
        self.last_tensors["obj_active_mem"] = obj_active_mem
        # [-1, 1]
        return unpack_sequence(obj_active_seq)

    @submodule("action")
    def trunk(
        self,
        action,
        dom_interest,
        action_status_indicators,
        obj_active,
        leaves_att,
        obj_att_input,
        ux_action_consensus,
    ):
        _obj_active = safe_bc(obj_active, action.field_index)
        _leaf_mask = safe_bc(leaves_att, action.leaf_dom_index)
        _obj_att_input = safe_bc(obj_att_input, action.field_dom_index)

        # main action space activity
        # leaf att, goal mask, ux action mask, value mask, enabled mask
        trunk = torch.cat(
            [
                dom_interest,
                action_status_indicators,
                _obj_active,
                _leaf_mask,
                _obj_att_input,
                ux_action_consensus,
            ],
            dim=1,
        )

        if SAFETY:
            assert global_max_pool(ux_action_consensus, action.batch).min().item() > 0
            assert global_max_pool(dom_interest, action.batch).min().item() > 0
        return trunk

    def action_trunk(self, action, trunk, ac_norm):
        action_trunk = global_max_pool(trunk, action.action_index)
        return torch.cat([ac_norm, action_trunk], dim=1)

    def action_votes(self, action_trunk):
        return self.actor_gate(action_trunk)

    def critic_x(
        self, field, trunk, obj_active, action_trunk, action_batch_idx, obj_active_mem
    ):
        # layers * directions, batch, hidden_size
        critic_near_completion = obj_active_mem[-1]

        # critic senses goal completion
        critic_active_steps = torch.tanh(global_add_pool(obj_active, field.batch))

        # critic senses trunk
        x_critic_input = action_trunk

        critic_mp = self.critic_mp_score(x_critic_input)
        critic_ap = self.critic_ap_score(x_critic_input)

        # max/add objective difficulty in batch
        critic_mp = torch.relu(global_max_pool(critic_mp, action_batch_idx))
        critic_ap = torch.tanh(global_add_pool(critic_ap, action_batch_idx))

        critic_x = torch.cat(
            [critic_mp, critic_ap, critic_active_steps, critic_near_completion], dim=1
        )
        return critic_x

    def action_batch_idx(self, action):
        return global_max_pool(action.batch.view(-1, 1), action.action_index).view(-1)

    def critic_value(self, critic_x):
        return self.critic_gate(critic_x)

    def forward(self, inputs, rnn_hxs, masks):
        from torch_geometric.data import Batch, Data

        assert isinstance(inputs, tuple), str(type(inputs))
        assert all(map(lambda x: isinstance(x, Batch), inputs))
        dom, objectives, objectives_projection, leaves, actions, history = inputs
        assert actions.edge_index is not None, str(inputs)
        self.last_tensors = {
            "dom": dom,
            "field": objectives,
            "dom_leaf": leaves,
            "dom_field": objectives_projection,
            "action": actions,
            "rnn_hxs": rnn_hxs,
            "masks": masks,
        }

        return (
            self.resolve(self.critic_value),
            (self.resolve(self.action_votes), self.resolve(self.action_batch_idx)),
            self.last_tensors["rnn_hxs"],
        )


class Encoder(torch.nn.Module):
    def __init__(self, model, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        if model in ["GAE"]:
            self.conv2 = GCNConv(2 * out_channels, out_channels)
        elif model in ["VGAE"]:
            self.conv_mu = GCNConv(2 * out_channels, out_channels)
            self.conv_logvar = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        if hasattr(self, "conv2"):
            return self.conv2(x, edge_index)
        else:
            return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


class WoBObservationEncoder(nn.Module):
    def __init__(self, in_dim=108, out_dim=64):
        nn.Module.__init__(self)
        self.conv1 = SAGEConv(in_dim, 128)
        self.conv2 = SAGEConv(128, out_dim)

    def forward(self, inputs, votes):
        edge_index = inputs.edge_index
        batch = inputs.batch

        x = torch.cat(
            [
                inputs[key]
                for key in [
                    "text",
                    "value",
                    "tag",
                    "classes",
                    "rx",
                    "ry",
                    "width",
                    "height",
                    "top",
                    "left",
                    "focused",
                ]
            ],
            dim=1,
        )
        x = torch.cat([x, votes], dim=1)
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_max_pool(x, batch)
        return x
