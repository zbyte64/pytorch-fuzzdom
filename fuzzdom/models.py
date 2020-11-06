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
    GlobalAttention,
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
from .nn import ResolveMixin, EdgeMask, EdgeAttrs


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


class DirectionalPropagation(ResolveMixin, nn.Module):
    def __init__(self, hidden_size, transitivity_size, mask_dim, K):
        super(DirectionalPropagation, self).__init__()
        self.dom_transitivity_fn = EdgeAttrs(hidden_size, transitivity_size)
        self.dom_edge_mask = EdgeMask(1 + transitivity_size, mask_dim, K)
        self.pos_edge_mask = EdgeMask(4 + transitivity_size, mask_dim, K)

    def spatial_mask(self, x, dom, projection, mask):
        spatial_edge_trans = self.dom_transitivity_fn(x, dom.spatial_edge_index)
        if projection is not dom:
            _spatial_edge_trans = safe_bc_edges(
                spatial_edge_trans, projection.br_spatial_edge_index
            )
        else:
            _spatial_edge_trans = spatial_edge_trans
        pos_mask, pos_mask_ew = self.pos_edge_mask(
            torch.cat([projection.spatial_edge_attr, _spatial_edge_trans], dim=1),
            mask,
            projection.spatial_edge_index,
        )
        self.export_value("spatial_edge_weights", pos_mask_ew)
        return pos_mask

    def dom_mask(self, x, dom, projection, mask):
        dom_edge_trans = self.dom_transitivity_fn(x, dom.dom_edge_index)
        if projection is not dom:
            _dom_edge_trans = safe_bc_edges(
                dom_edge_trans, projection.br_dom_edge_index
            )
        else:
            _dom_edge_trans = dom_edge_trans
        dom_mask, dom_mask_ew = self.dom_edge_mask(
            torch.cat([projection.dom_edge_attr, _dom_edge_trans], dim=1),
            mask,
            projection.dom_edge_index,
        )
        self.export_value("dom_edge_weights", dom_mask_ew)
        return dom_mask

    def forward(self, x, dom, projection, mask):
        r = self.start_resolve(
            {"x": x, "dom": dom, "mask": mask, "projection": projection}
        )
        pos_mask = r["spatial_mask"]
        dom_mask = r["dom_mask"]
        return torch.max(mask, torch.max(pos_mask, dom_mask))


class SoftActionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.default_action = 1  # input goal
        self.action_words = [
            ["click", "submit", "target", "focus"],
            ["select", "text", "noun", "username", "password"],
            ["copy"],
            ["paste"],
        ]
        action_size = len(self.action_words)
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
        self.vote_fn = nn.Sequential(
            init_ot(nn.Linear(action_size, action_size)), nn.Softmax(dim=1),
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

        return self.vote_fn(batchwise_action_sims)


def autoencoder_x(dom):
    return torch.cat(
        [
            dom.text,
            dom.value,
            dom.tag,
            dom.classes,
            dom.radio_value,
            dom.rx,
            dom.ry,
            dom.width,
            dom.height,
            dom.top,
            dom.left,
            dom.depth,
        ],
        dim=1,
    )


def domain(target_domain=None):
    def f(func):
        func.__target_domain__ = target_domain
        return func

    return f


class GNNBase(ResolveMixin, NNBase):
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

        action_one_hot_size = 4  # 5 to enable wait
        query_input_dim = 10 + 2 * text_embed_size
        if dom_encoder:
            query_input_dim += dom_encoder.out_channels
        self.attr_norm = nn.BatchNorm1d(text_embed_size)
        self.action_decoder = SoftActionDecoder()
        self.global_conv = SAGEConv(query_input_dim, hidden_size - query_input_dim)
        self.global_dropout = nn.Dropout(p=0.0)

        # paste  & copy have equal weight
        dom_ux_action_fn_size = action_one_hot_size - 1
        self.dom_ux_action_fn = nn.Sequential(
            init_xn(nn.Linear(hidden_size, dom_ux_action_fn_size), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(dom_ux_action_fn_size, dom_ux_action_fn_size), "linear"),
            nn.Softmax(dim=1),
        )
        transitivity_size = 4
        self.leaf_prop = DirectionalPropagation(
            hidden_size, transitivity_size, mask_dim=1, K=5
        )
        self.goal_prop = DirectionalPropagation(
            hidden_size, transitivity_size, mask_dim=1, K=5
        )
        self.value_prop = DirectionalPropagation(
            hidden_size, transitivity_size, mask_dim=1, K=5
        )

        self.dom_description_fn = nn.Sequential(
            init_xn(nn.Linear(hidden_size, text_embed_size), "relu"),
            nn.ReLU(),
            init_xu(nn.Linear(text_embed_size, text_embed_size), "tanh"),
            nn.Tanh(),
        )
        # attr similarities
        self.attr_similarity_size = attr_similarity_size = 10
        # [ goal att, value ]
        self.dom_objective_attr_fn = nn.Sequential(
            init_xn(nn.Linear(hidden_size, 2 * attr_similarity_size), "relu"),
            nn.ReLU(),
            init_xn(
                nn.Linear(2 * attr_similarity_size, 2 * attr_similarity_size), "sigmoid"
            ),
            nn.Sigmoid(),
        )
        # [ enabled ]
        self.dom_disabled_fn = nn.Sequential(
            init_xn(nn.Linear(hidden_size, 8), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(8, 1), "sigmoid"),
            nn.Sigmoid(),
        )
        self.dom_disabled_prop = DirectionalPropagation(
            hidden_size, transitivity_size, mask_dim=1, K=5,
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
        self.ux_action_needs_value_fn = nn.Sequential(
            init_xn(nn.Linear(action_one_hot_size, 1), "sigmoid"), nn.Sigmoid(),
        )
        self.dom_needs_value_fn = nn.Sequential(
            init_xn(nn.Linear(hidden_size, 8), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(8, 1), "sigmoid"),
            nn.Sigmoid(),
        )
        self.dom_disabled_value_fn = nn.Sequential(
            init_xn(nn.Linear(hidden_size, 8), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(8, 1), "sigmoid"),
            nn.Sigmoid(),
        )
        self.dom_disabled_value_prop = DirectionalPropagation(
            hidden_size, transitivity_size, mask_dim=1, K=5,
        )
        # has_value, order, is last, has focus, ux action
        objective_active_size = 4 + action_one_hot_size
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
            bidirectional=True,
        )
        self.active_fn = nn.Sequential(
            init_xn(nn.Linear(objective_active_size * 2, 1), "linear")
        )
        self.trunk_norm = InstanceNorm(1)
        trunk_size = 3
        self.actor_gate = nn.Sequential(
            init_ones(nn.Linear(trunk_size, 1), "relu"), nn.ReLU(),
        )

        critic_embed_size = 16
        # ac norm, active, has value
        critic_size = 3
        self.critic_mp_score = nn.Sequential(
            init_xu(nn.Linear(critic_size, critic_size), "relu"), nn.ReLU(),
        )
        self.critic_ap_score = nn.Sequential(
            init_xu(nn.Linear(critic_size, critic_size), "relu"), nn.ReLU(),
        )
        critic_gate_size = (
            2 * critic_size + 1 * objective_active_size
        )  # active steps, near_completion
        self.critic_gate = nn.Sequential(
            init_xu(nn.Linear(critic_gate_size, critic_embed_size), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(critic_embed_size, 1), "linear"),
        )

    @domain("dom")
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
                dom.tampered,
                dom.depth,
            ],
            dim=1,
        ).clamp(-1, 1)

    @domain("dom")
    def encoded_x(self, dom):
        if not self.dom_encoder:
            return None
        with torch.no_grad():
            return torch.tanh(self.dom_encoder(autoencoder_x(dom), dom.dom_edge_index,))

    @domain("dom")
    def full_x(self, dom, x, encoded_x):
        if encoded_x is not None:
            x = torch.cat([x, encoded_x], dim=1)

        _add_x = self.global_dropout(self.global_conv(x, dom.spatial_edge_index))

        full_x = torch.cat([x, _add_x], dim=1)
        return full_x

    @domain("dom_leaf")
    def leaves_att(self, dom, dom_leaf, full_x):
        leaf_t_mask = dom_leaf.mask.type(torch.float)
        return self.leaf_prop(full_x, dom, dom_leaf, leaf_t_mask)

    @domain("dom")
    def dom_ux_action(self, full_x):
        # infer action from dom nodes
        dom_ux_action = self.dom_ux_action_fn(full_x)
        # for dom action input == paste
        return torch.cat([dom_ux_action[:, 0:], dom_ux_action[:, 1].view(-1, 1)], dim=1)

    @domain("dom_leaf")
    def leaf_ux_action(self, dom_ux_action, dom_leaf):
        return global_max_pool(
            safe_bc(dom_ux_action, dom_leaf.dom_index) * dom_leaf.mask,
            dom_leaf.leaf_index,
        )

    @domain("dom_field")
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
            if not (
                global_max_pool(obj_tag_similarities.max(dim=1).values, dom_field.batch)
                .min()
                .item()
                > 0
            ):
                print("Warning: obj_tag_similarities didn't have positive values")
                print(obj_tag_similarities)
                print(field.query)
                print(field.key)
                print(
                    global_max_pool(
                        obj_tag_similarities.max(dim=1).values, dom_field.batch
                    )
                )
                assert False
        return obj_tag_similarities

    @domain("dom_field")
    def dom_obj_attr(self, dom_field, full_x):
        """
        Compute attribute affinity to goal target or completion
        Returns: [:,:,0] = goal target. [:,:,1] = goal completion.
        [0,1]
        """
        dom_obj_attr = self.dom_objective_attr_fn(full_x).view(
            full_x.shape[0], self.attr_similarity_size, 2
        )
        # dom_obj_attr = F.softmax(dom_obj_attr, dim=2)
        return safe_bc(dom_obj_attr, dom_field.dom_index)

    @domain("dom_field")
    def dom_obj_int(self, dom_obj_attr, obj_tag_similarities):
        # ux action tag sensitivities
        return torch.relu(
            torch.einsum("bad,ba->bad", dom_obj_attr, obj_tag_similarities)
        )

    @domain("dom_field")
    def obj_att_input(self, dom, dom_field, full_x, dom_obj_int):
        obj_att_input = torch.relu(
            (dom_obj_int[:, :, 0]).max(dim=1, keepdim=True).values
        )

        if SAFETY:
            assert global_max_pool(obj_att_input, dom_field.batch).min().item() > 0

        return self.goal_prop(full_x, dom, dom_field, obj_att_input)

    @domain("field")
    def obj_ux_action(self, field):
        return self.objective_ux_action_fn(field.key)

    @domain("action")
    def ux_action_consensus(self, action, obj_ux_action, leaf_ux_action):
        _obj_ux_action = safe_bc(
            obj_ux_action.view(-1, 1), action.field_ux_action_index
        )

        _leaf_ux_action = safe_bc(
            leaf_ux_action.view(-1, 1), action.leaf_ux_action_index
        )
        return _leaf_ux_action * _obj_ux_action

    @domain("dom")
    def leaf_selector(
        self,
        dom,
        field,
        dom_field,
        dom_ux_action,
        obj_ux_action,
        disabled_mask,
        obj_att_input,
    ):
        ux_action_consensus = global_max_pool(
            safe_bc(global_max_pool(obj_ux_action, field.batch), dom_field.field_index)
            * safe_bc(dom_ux_action, dom_field.dom_index),
            dom_field.dom_index,
        )
        obj_int = global_max_pool(obj_att_input, dom_field.dom_index)
        active_mask = 1 - disabled_mask
        return (
            ux_action_consensus.max(dim=1, keepdim=True).values * active_mask * obj_int
        )

    @domain("action")
    def dom_interest(self, action, leaves_att, disabled_mask, ux_action_consensus):
        _leaf_mask = safe_bc(leaves_att, action.leaf_dom_index)
        _disabled_mask = safe_bc(disabled_mask, action.dom_index)
        if SAFETY:
            assert global_max_pool(_leaf_mask, action.batch).min().item() > 0

        return _leaf_mask * ux_action_consensus * _disabled_mask

    @domain("action")
    def action_consensus(self, action, dom_interest, obj_att_input):
        _obj_att_input = safe_bc(obj_att_input, action.field_dom_index)
        action_consensus = dom_interest * _obj_att_input
        if SAFETY:
            assert global_max_pool(action_consensus, action.batch).min().item() > 0
        return action_consensus

    @domain("action_index")
    def ac_value(self, action, ac_norm):
        """
        scale each action value relative to other action options within the same goal
        """
        return global_max_pool(ac_norm, action.action_index)

    @domain("action")
    def ac_norm(self, action, action_consensus):
        """
        normalize action_consensus across goals
        """
        return torch.relu(self.trunk_norm(action_consensus, action.field_index))

    @domain("action")
    def dom_interest_norm(self, action, dom_interest, ac_value):
        _ac_value = safe_bc(ac_value, action.action_index)
        return dom_interest * _ac_value

    @domain("dom")
    def dom_disabled_value_mask(self, dom, full_x):
        dom_disabled = self.dom_disabled_value_fn(full_x)
        # invert after propagation
        return 1 - self.dom_disabled_value_prop(full_x, dom, dom, dom_disabled)

    @domain("dom_field")
    def value_mask(
        self, dom, dom_field, full_x, dom_obj_int, dom_disabled_value_mask,
    ):
        """
        compute objective completeness with goal word similarities
        [0 - 1]
        """
        obj_value_mask = (dom_obj_int[:, :, 1]).max(dim=1, keepdim=True).values
        _dom_value_mask = safe_bc(dom_disabled_value_mask, dom_field.dom_index)
        non_value = self.dom_needs_value_fn(full_x)
        _non_value = safe_bc(non_value, dom_field.dom_index)
        value_mask = (
            torch.cat([obj_value_mask, _non_value], dim=1)
            .max(dim=1, keepdim=True)
            .values
        )
        value_mask = (
            self.value_prop(full_x, dom, dom_field, value_mask) * _dom_value_mask
        )
        return value_mask

    @domain("dom")
    def disabled_mask(self, dom, full_x):
        """
        determine if a node is active/interesting
        """
        mask = self.dom_disabled_fn(full_x)
        # invert after positive propagation
        return 1 - self.dom_disabled_prop(full_x, dom, dom, mask)

    @domain("action")
    def action_status_indicators(
        self, dom, action, dom_interest_norm, value_mask, obj_ux_action
    ):
        """
        determine if a node supplies a goal value (or doesn't need to)
        """
        _value_mask = safe_bc(value_mask, action.field_dom_index)
        non_value = self.ux_action_needs_value_fn(obj_ux_action)
        _non_value = safe_bc(non_value, action.field_index)
        v = torch.cat([_value_mask, _non_value], dim=1)
        return v.max(dim=1, keepdim=True).values * dom_interest_norm

    @domain("field")
    def obj_indicator_order(
        self,
        dom,
        field,
        dom_field,
        action,
        dom_interest_norm,
        action_status_indicators,
        obj_ux_action,
        full_x,
        rnn_hxs,
        masks,
    ):
        """
        Input for determining the active step
        """
        # has the value been supplied?
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
                global_max_pool(dom_obj_focus * dom_interest_norm, action.field_index),
                obj_ux_action,
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
            self.export_value("rnn_hxs", rnn_hxs)
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

    @domain("field")
    def obj_active(self, field, obj_indicator_order):
        obj_ind_seq = pack_as_sequence(obj_indicator_order, field.batch)
        obj_active_seq, obj_active_mem = self.objective_active(obj_ind_seq)
        self.export_value("obj_active_mem", obj_active_mem)
        # [0, 1]
        return softmax(self.active_fn(unpack_sequence(obj_active_seq)), field.batch)

    @domain("action_index")
    def critic_trunk(
        self, action, ac_norm, action_status_indicators, obj_active,
    ):
        _obj_active = safe_bc(obj_active, action.field_index)

        trunk = torch.cat([action_status_indicators, _obj_active, ac_norm,], dim=1,)
        return global_max_pool(trunk, action.action_index)

    @domain("action_index")
    def action_trunk(self, action, ac_norm, action_status_indicators, obj_active):
        _obj_active = safe_bc(obj_active, action.field_index)
        trunk = torch.cat(
            [ac_norm, ac_norm - action_status_indicators, _obj_active * ac_norm], dim=1
        )
        return global_max_pool(trunk, action.action_index)

    def action_votes(self, action_trunk):
        return self.actor_gate(action_trunk)

    def critic_x(
        self, field, action, critic_trunk, obj_active, action_batch_idx, obj_active_mem
    ):
        # layers * directions, batch, hidden_size
        critic_near_completion = obj_active_mem[-1]

        critic_mp = self.critic_mp_score(critic_trunk)
        critic_ap = self.critic_ap_score(critic_trunk)

        # max/add objective difficulty in batch
        critic_mp = torch.relu(global_max_pool(critic_mp, action_batch_idx))
        critic_ap = torch.tanh(global_add_pool(critic_ap, action_batch_idx))

        critic_x = torch.cat([critic_mp, critic_ap, critic_near_completion], dim=1)
        return critic_x

    def action_batch_idx(self, action):
        return global_max_pool(action.batch.view(-1, 1), action.action_index).view(-1)

    def critic_value(self, critic_x):
        return self.critic_gate(critic_x)

    def forward(self, inputs, rnn_hxs, masks):
        from torch_geometric.data import Batch, Data

        assert isinstance(inputs, tuple), str(type(inputs))
        assert all(map(lambda x: isinstance(x, Batch), inputs))
        dom, objectives, objectives_projection, leaves, actions, *history = inputs
        assert actions.edge_index is not None, str(inputs)
        r = self.start_resolve(
            {
                "dom": dom,
                "field": objectives,
                "dom_leaf": leaves,
                "dom_field": objectives_projection,
                "action": actions,
                "rnn_hxs": rnn_hxs,
                "masks": masks,
            }
        )

        return (
            r["critic_value"],
            (r["action_votes"], r["action_batch_idx"]),
            r["rnn_hxs"],
        )


class Instructor(GNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # tag and classes already included in full_x
        self.instructor_size = self.hidden_size + self.text_embed_size * 2
        self.key_fn = GlobalAttention(
            nn.Sequential(
                init_xn(nn.Linear(self.instructor_size, self.text_embed_size), "relu"),
                nn.ReLU(),
                init_xn(nn.Linear(self.text_embed_size, 1), "relu"),
                nn.ReLU(),
            )
        )
        self.query_fn = GlobalAttention(
            nn.Sequential(
                init_xn(nn.Linear(self.instructor_size, self.text_embed_size), "relu"),
                nn.ReLU(),
                init_xn(nn.Linear(self.text_embed_size, 1), "relu"),
                nn.ReLU(),
            )
        )
        from .domx import short_embed

        self.target_words = [
            "click",
            "submit",
            "target",
            "focus",
            "select",
            "copy",
            "paste",
        ]
        self.other_values = torch.cat(
            list(map(lambda x: torch.from_numpy(short_embed(x)), self.target_words,))
        ).view(1, -1)
        self.key_selection_size = len(self.target_words) + 4
        self.key_softmax_fn = nn.Sequential(
            init_xn(nn.Linear(self.instructor_size, self.text_embed_size), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(self.text_embed_size, self.key_selection_size)),
            # nn.ReLU(),
        )
        self.query_softmax_fn = nn.Sequential(
            init_xn(nn.Linear(self.instructor_size, self.text_embed_size), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(self.text_embed_size, 4)),
            # nn.ReLU(),
        )

    def field(self, dom, _field, full_x):
        batch_size = _field.batch.shape[0]
        device = full_x.device
        # tag and classes already included in the front
        fuller_x = torch.cat([dom.text, dom.value, full_x], dim=1)
        key_softmax_x = global_max_pool(self.key_softmax_fn(fuller_x), dom.batch)
        key_softmax = F.softmax(key_softmax_x, dim=1).repeat_interleave(
            self.text_embed_size, dim=1
        )
        key = self.key_fn(fuller_x, dom.batch)[:, : self.text_embed_size * 4]
        key = torch.cat(
            [key, self.other_values.to(device).repeat(batch_size, 1)], dim=1
        )

        _field.key = (
            (key * key_softmax)
            .view(batch_size, self.text_embed_size, self.key_selection_size)
            .sum(dim=2)
        )
        assert _field.key.shape == _field.query.shape, str(
            (_field.key.shape, _field.query.shape)
        )
        query_softmax_x = global_max_pool(self.query_softmax_fn(fuller_x), dom.batch)
        query_softmax = F.softmax(query_softmax_x, dim=1).repeat_interleave(
            self.text_embed_size, dim=1
        )
        query = self.query_fn(fuller_x, dom.batch)[:, : self.text_embed_size * 4]
        _field.query = (
            (query * query_softmax).view(batch_size, self.text_embed_size, 4).sum(dim=2)
        )
        with torch.no_grad():
            if (
                _field.query.clone().detach().requires_grad_(False).max().min().item()
                <= 0
            ):
                print(query_softmax_x)
                print(query_softmax)
                print(query)
                print(_field.query)
                assert False
        return _field

    def forward(self, inputs, rnn_hxs, masks):
        from torch_geometric.data import Batch, Data

        assert isinstance(inputs, tuple), str(type(inputs))
        assert all(map(lambda x: isinstance(x, Batch), inputs))
        dom, objectives, objectives_projection, leaves, actions, *_ = inputs
        assert actions.edge_index is not None, str(inputs)
        self.start_resolve(
            {
                "dom": dom,
                "_field": objectives,
                "dom_leaf": leaves,
                "dom_field": objectives_projection,
                "action": actions,
                "rnn_hxs": rnn_hxs,
                "masks": masks,
            }
        )

        return (
            self.resolve_value(self.critic_value),
            (
                self.resolve_value(self.action_votes),
                self.resolve_value(self.action_batch_idx),
            ),
            self.resolve_value(lambda rnn_hxs: rnn_hxs),
        )


class Encoder(nn.Module):
    def __init__(self, model, in_channels, out_channels):
        super().__init__()
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
