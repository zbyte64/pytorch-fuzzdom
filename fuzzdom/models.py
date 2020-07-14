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
    def __init__(self, input_dim, out_dim=1):
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

        action_one_hot_size = 4  # 5
        query_input_dim = 6 + 3 * text_embed_size
        if dom_encoder:
            query_input_dim += dom_encoder.out_channels
        self.attr_norm = nn.BatchNorm1d(text_embed_size)
        self.action_decoder = FixedActionDecoder()
        self.global_conv = SAGEConv(query_input_dim, hidden_size - query_input_dim)

        self.dom_ux_action_fn = nn.Sequential(
            # paste  & copy have equal weight
            init_xn(nn.Linear(hidden_size, action_one_hot_size - 1), "linear"),
            nn.Softmax(dim=1),
        )
        self.dom_transitivity_fn = EdgeAttrs(hidden_size)
        # attr similarities
        self.attr_similarity_size = attr_similarity_size = 8
        self.objective_pos_conv = EdgeMask(4 + 1, 1)
        # [ goal att, value ]
        self.dom_objective_attr_fn = nn.Sequential(
            init_xn(nn.Linear(hidden_size, 2 * attr_similarity_size), "sigmoid"),
            nn.Sigmoid(),
        )
        # [ enabled ]
        self.dom_objective_enable_fn = nn.Sequential(
            init_xn(nn.Linear(hidden_size, 1), "sigmoid"), nn.Sigmoid()
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
        objective_indicator_size = 1
        # global_max_pool([action consensus, enabled, value similarity])
        self.objective_looks_complete = nn.GRU(
            input_size=3, hidden_size=8, num_layers=1
        )
        # + pos + focus + global focus
        objective_active_size = (
            action_one_hot_size + objective_indicator_size + 2 + 1 + 1
        )
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
            hidden_size=8,
            num_layers=2,
            # bidirectional=True,
        )
        # norm active step selection
        self.objective_active_norm = InstanceNorm(1)
        # [ux action, norm consensus * active, dev] + (consensus - avg), active*2
        trunk_size = action_one_hot_size + 5
        # norm consensus
        self.trunk_norm = InstanceNorm(1)
        self.actor_gate = nn.Sequential(
            init_xn(nn.Linear(trunk_size, trunk_size), "relu"),
            nn.ReLU(),
            init_ones(nn.Linear(trunk_size, 1), "relu"),
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
        self.graph_size_norm = GraphSizeNorm()
        critic_gate_size = (
            2 * critic_embed_size + 1 + 1
        )  # active steps, near_completion
        self.critic_gate = nn.Sequential(
            init_xu(nn.Linear(critic_gate_size, critic_embed_size), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(critic_embed_size, 1), "linear"),
        )

    def forward(self, inputs, rnn_hxs, masks):
        from torch_geometric.data import Batch, Data

        assert isinstance(inputs, tuple), str(type(inputs))
        assert all(map(lambda x: isinstance(x, Batch), inputs))
        dom, objectives, objectives_projection, leaves, actions, history = inputs
        assert actions.edge_index is not None, str(inputs)
        self.last_tensors = {}

        x = torch.cat(
            [
                dom.tag,
                dom.classes,
                dom.text,
                dom.rx,
                dom.ry,
                dom.width * dom.height,
                dom.top * dom.left,
                dom.focused,
                dom.depth,
            ],
            dim=1,
        ).clamp(-1, 1)

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

        _add_x = self.global_conv(x, full_edges(dom.dom_edge_index))

        x = torch.cat([x, _add_x], dim=1)
        self.last_tensors["conv_x"] = _add_x
        full_x = x

        # leaves are targetable elements
        leaf_t_mask = leaves.mask.type(torch.float)
        # compute a leaf dependent dom feats to indicate relevent nodes
        dom_edge_trans = self.dom_transitivity_fn(full_x, dom.edge_index)
        assert dom.edge_index.shape[1] == dom_edge_trans.shape[0]
        dom_edge_trans = safe_bc_edges(dom_edge_trans, leaves.br_dom_edge_index)
        leaves_att, leaves_ew = self.objective_pos_conv(
            torch.cat([leaves.edge_attr, dom_edge_trans], dim=1),
            leaf_t_mask,
            leaves.edge_index,
        )
        leaves_att = torch.max(leaves_att, leaf_t_mask)
        self.last_tensors["leaves_edge_weights"] = leaves_ew

        # infer action from dom nodes
        dom_ux_action = self.dom_ux_action_fn(full_x)
        # for dom action input == paste
        dom_ux_action = torch.cat(
            [dom_ux_action[:, 0:], dom_ux_action[:, 1].view(-1, 1)], dim=1
        )
        leaf_ux_action = global_max_pool(
            safe_bc(dom_ux_action, leaves.dom_index) * leaves.mask, leaves.leaf_index
        )

        # compute an objective dependent dom feats
        # start with word embedding similarities
        obj_sim = lambda tag, obj: F.cosine_similarity(
            safe_bc(dom[tag], objectives_projection.dom_index),
            safe_bc(objectives[obj], objectives_projection.field_index),
            dim=1,
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
                ],
                dim=1,
            )
        )
        if SAFETY:
            assert (
                global_max_pool(
                    obj_tag_similarities.max(dim=1)[0], objectives_projection.batch
                )
                .min()
                .item()
                > 0
            )
        # infer action from query key
        obj_ux_action = self.objective_ux_action_fn(objectives.key)
        # infer interested dom attrs
        # [:,:,0] = goal target. [:,:,2] = goal completion. ; odds are cuttoff
        dom_obj_attr = self.dom_objective_attr_fn(full_x).view(
            full_x.shape[0], self.attr_similarity_size, 2
        )
        dom_obj_attr__op = safe_bc(dom_obj_attr, objectives_projection.dom_index)
        dom_obj_int = torch.sigmoid(dom_obj_attr__op[:, :, 0])
        # ux action tag sensitivities
        obj_att_input = torch.relu(
            (dom_obj_int * obj_tag_similarities).max(dim=1, keepdim=True).values
        )

        if SAFETY:
            assert (
                global_max_pool(obj_att_input, objectives_projection.batch).min().item()
                > 0
            )

        self.last_tensors["dom_obj_attr"] = dom_obj_attr
        self.last_tensors["obj_att_input"] = obj_att_input
        self.last_tensors["obj_tag_similarities"] = obj_tag_similarities

        # project into main action space
        _obj_ux_action = safe_bc(
            obj_ux_action.view(-1, 1), actions.field_ux_action_index
        )

        _leaf_ux_action = safe_bc(
            leaf_ux_action.view(-1, 1), actions.leaf_ux_action_index
        )
        _leaf_mask = safe_bc(leaves_att, actions.leaf_dom_index)

        if SAFETY:
            assert global_max_pool(_leaf_mask, actions.batch).min().item() > 0

        ux_action_consensus = _leaf_ux_action * _obj_ux_action
        dom_interest = _leaf_mask * safe_bc(obj_att_input, actions.field_dom_index)
        action_consensus = ux_action_consensus * dom_interest

        # norm activity within each goal
        trunk_norm = self.trunk_norm(action_consensus, actions.field_index)
        self.last_tensors["trunk_norm"] = trunk_norm

        # begin determining active step
        # compute objective completeness indicators from DOM
        x_dom_obj_enabled = (
            safe_bc(self.dom_objective_enable_fn(full_x), actions.dom_index)
            * _leaf_mask
        )
        # compute objective completeness with goal word similarities
        # [0 - 1]
        dom_obj_comp_attr = dom_obj_attr__op[:, :, 1]
        dom_obj_comp = (
            safe_bc(
                (dom_obj_comp_attr * obj_tag_similarities)
                .max(dim=1, keepdim=True)
                .values,
                actions.field_dom_index,
            )
            * _leaf_mask
        )

        self.last_tensors["x_dom_obj_enabled"] = x_dom_obj_enabled
        self.last_tensors["dom_obj_comp"] = dom_obj_comp

        action_indicators = global_max_pool(
            torch.cat([action_consensus, dom_obj_comp, x_dom_obj_enabled], dim=1),
            actions.action_index,
        )
        action_ind_seq = pack_as_sequence(
            action_indicators,
            global_max_pool(actions.field_index, actions.action_index),
        )
        # flat fields
        _, action_active_mem = self.objective_looks_complete(action_ind_seq)
        # [-1, 1]
        obj_indicator = action_active_mem[0, objectives.index, -1:]

        # determine if a goal has focus
        goal_focus = safe_bc(dom.focused, actions.dom_index)
        goal_focus = global_max_pool(goal_focus * action_consensus, actions.field_index)
        has_focus = global_max_pool(goal_focus, objectives.batch)

        # pack on order embed and goal focus
        obj_indicator_order = torch.cat(
            [
                safe_bc(has_focus, objectives.batch),
                goal_focus,
                obj_ux_action,
                obj_indicator,
                1 - objectives.order,
                objectives.is_last,
            ],
            dim=1,
        )
        assert obj_indicator_order.shape[1] == 9, str(obj_indicator_order.shape)

        if self.is_recurrent:
            goal_dom_input = torch.cat(
                [
                    safe_bc(x, objectives_projection.dom_index),
                    safe_bc(obj_indicator_order, objectives_projection.field_index),
                ],
                dim=1,
            )
            goal_dom_encoded = self.goal_dom_encoder(
                goal_dom_input, objectives_projection.dom_edge_index
            )
            goal_dom_flat = global_max_pool(
                goal_dom_encoded, objectives_projection.batch
            )
            curr_state, rnn_hxs = self._forward_gru(goal_dom_flat, rnn_hxs, masks)
            state_indicator_input = torch.cat(
                [goal_dom_input, safe_bc(curr_state, objectives_projection.batch)],
                dim=1,
            )
            state_indicator = global_max_pool(
                self.state_indicator(state_indicator_input),
                objectives_projection.field_index,
            )
            obj_indicator_order = torch.cat(
                [state_indicator, obj_indicator_order], dim=1
            )

        obj_ind_seq = pack_as_sequence(obj_indicator_order, objectives.batch)
        obj_active_seq, obj_active_mem = self.objective_active(obj_ind_seq)
        # [-1, 1]
        obj_active_share = unpack_sequence(obj_active_seq)[:, -1:]
        # [0, 2]
        obj_active = (1 + obj_active_share) * (1 - obj_indicator)
        _obj_active = safe_bc(obj_active, actions.field_index)

        self.last_tensors["obj_indicator"] = obj_indicator
        self.last_tensors["obj_active"] = obj_active

        # main action space activity
        trunk_dev, trunk_med = scatter_std_dev(action_consensus, actions.field_index)
        trunk_dev = trunk_dev[actions.field_index]
        trunk_med = trunk_med[actions.field_index]
        trunk_ap = _obj_active * action_consensus
        trunk = torch.cat(
            [
                trunk_ap,
                trunk_ap - trunk_med,
                trunk_ap * trunk_norm,
                _obj_active * trunk_norm,
                trunk_dev,
                actions.action_one_hot,
            ],
            dim=1,
        )

        if SAFETY:
            assert global_max_pool(ux_action_consensus, actions.batch).min().item() > 0
            assert global_max_pool(dom_interest, actions.batch).min().item() > 0
            assert global_max_pool(action_consensus, actions.batch).min().item() > 0

        self.last_tensors["action_consensus"] = action_consensus
        self.last_tensors["ux_action_consensus"] = ux_action_consensus
        self.last_tensors["dom_interest"] = dom_interest
        # gather the batch ids for the votes
        action_batch_idx = global_max_pool(
            actions.batch.view(-1, 1), actions.action_index
        ).view(-1)
        action_votes = global_max_pool(self.actor_gate(trunk), actions.action_index)
        action_idx = global_max_pool(actions.action_idx, actions.action_index)

        self.last_tensors["leaves_att"] = leaves_att
        self.last_tensors["obj_ux_action"] = obj_ux_action
        self.last_tensors["dom_ux_action"] = dom_ux_action

        # layers * directions, batch, hidden_size
        critic_near_completion = obj_active_mem[-1, :, -1:]

        # critic senses goal completion
        critic_active_steps = torch.tanh(
            global_add_pool(obj_active_share, objectives.batch) - 1
        )

        # critic senses trunk
        x_critic_input = trunk

        critic_mp = self.critic_mp_score(x_critic_input)
        critic_ap = self.critic_ap_score(x_critic_input)

        # max/add objective difficulty in batch
        critic_mp = torch.relu(global_max_pool(critic_mp, actions.batch))
        critic_ap = torch.tanh(global_add_pool(critic_ap, actions.batch))

        critic_x = torch.cat(
            [critic_mp, critic_ap, critic_active_steps, critic_near_completion], dim=1
        )
        critic_value = self.critic_gate(critic_x)

        self.last_tensors["x"] = x
        self.last_tensors["trunk"] = trunk
        self.last_tensors["action_votes"] = action_votes
        self.last_tensors["critic_x"] = critic_x

        return (critic_value, (action_votes, action_batch_idx), rnn_hxs)


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
