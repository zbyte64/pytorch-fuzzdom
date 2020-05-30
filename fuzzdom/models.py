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

from a2c_ppo_acktr.model import NNBase
from a2c_ppo_acktr.utils import init

from .functions import *


# do extra checks, useful for validating model changes
SAFETY = True

if SAFETY:
    torch.set_printoptions(profile="full")
    torch.autograd.set_detect_anomaly(True)


def safe_bc(x, b, saturated=True):
    """
    Doas a matrix broadcast but checks that the mask size is appropriate
    """
    m = None
    if SAFETY:
        assert len(x.shape) > 1
        assert len(b.shape) == 1, "Mask should be 1D"
        m = b.max().item() + 1
        if saturated:
            assert x.shape[0] == m, f"Incomplete/Wrong mask {x.shape[0]} != {m}"
        else:
            # mask takes a subsample
            assert x.shape[0] >= m, f"Incomplete/Wrong mask {x.shape[0]} != {m}"
    return x[b]


class AdditiveMask(nn.Module):
    def __init__(self, input_dim, mask_dim=1, K=5, bias=True):
        super(AdditiveMask, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1e-3))
        self.x_fn = nn.Sequential(
            init_ot(nn.Linear(input_dim, input_dim), "tanh"), nn.Tanh()
        )
        self.conv = APPNP(K=K, alpha=self.alpha)
        self.K = K
        if bias:
            self.bias = nn.Parameter(torch.ones(mask_dim) * -3)
        else:
            sekf.bias = None

    def forward(self, x, mask, edge_index):
        x = self.x_fn(x)
        edge_weights = torch.relu(
            F.cosine_similarity(x[edge_index[0]], x[edge_index[1]])
        ).view(-1)
        fill = torch.relu(mask)
        fill = self.conv(fill, edge_index, edge_weights)
        if self.bias is not None:
            fill = fill - F.softplus(self.bias)
        fill = torch.tanh(fill)
        return fill, edge_weights


class EdgeMask(nn.Module):
    def __init__(self, edge_dim, mask_dim=1, K=5, bias=True):
        super(EdgeMask, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1e-3))
        self.edge_fn = nn.Sequential(
            init_xu(nn.Linear(edge_dim, edge_dim), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(edge_dim, 1), "sigmoid"),
            nn.Sigmoid(),
        )
        self.conv = APPNP(K=K, alpha=self.alpha)
        self.K = K
        if bias:
            self.bias = nn.Parameter(torch.ones(mask_dim) * -3)
        else:
            sekf.bias = None

    def forward(self, edge_attr, mask, edge_index):
        edge_weights = self.edge_fn(edge_attr).view(-1)
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
        self.dom_tag = nn.Sequential(
            init_xn(nn.Linear(text_embed_size, action_one_hot_size), "relu"), nn.ReLU()
        )
        self.dom_classes = nn.Sequential(
            init_xn(nn.Linear(text_embed_size, action_one_hot_size), "relu"), nn.ReLU()
        )
        self.dom_text = nn.Sequential(
            init_xn(nn.Linear(text_embed_size, action_one_hot_size), "relu"), nn.ReLU()
        )
        self.dom_fn = nn.Sequential(
            init_xu(nn.Linear(query_input_dim, hidden_size - query_input_dim), "tanh"),
            nn.Tanh(),
        )
        self.global_alpha = nn.Parameter(torch.tensor(0.1))
        self.global_conv = APPNP(K=7, alpha=self.global_alpha, bias=False)

        self.dom_ux_action_decoder_weight = nn.Parameter(
            torch.ones(3, action_one_hot_size) * 0.5
        )
        self.dom_ux_action_fn = nn.Sequential(
            init_ot(nn.Linear(action_one_hot_size, action_one_hot_size), "linear"),
            nn.Softmax(dim=1),
        )
        self.leaves_conv = AdditiveMask(hidden_size, 1, K=7)
        # attr similarities
        self.attr_similarity_size = attr_similarity_size = 8
        self.objective_dom_tag = nn.Sequential(
            init_xu(nn.Linear(text_embed_size, text_embed_size), "relu"), nn.ReLU()
        )
        self.objective_dom_class = nn.Sequential(
            init_xu(nn.Linear(text_embed_size, text_embed_size), "relu"), nn.ReLU()
        )
        self.objective_conv = AdditiveMask(hidden_size, 1, K=7)
        self.objective_pos_conv = EdgeMask(4, 1)
        self.objective_ux_action_attr = nn.Sequential(
            init_xn(nn.Linear(action_one_hot_size, 3 * attr_similarity_size), "linear")
        )
        self.objective_comp_fn = nn.Sequential(
            init_xn(nn.Linear(action_one_hot_size, attr_similarity_size), "relu"),
            nn.ReLU(),
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
        # (consensus, softmax, norm) * ([1, consensus])
        trunk_size = 6
        self.trunk_norm = InstanceNorm(1)
        self.actor_gate = nn.Sequential(
            init_xu(nn.Linear(trunk_size, trunk_size), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(trunk_size, 1), "relu"),
            nn.ReLU(),
        )

        critic_dom_embed_size = 16
        critic_size = (
            attr_similarity_size
            + 2 * action_one_hot_size
            + 1  # obj att
            + 1  # ux action agreement
            + critic_dom_embed_size  # critic conv
        )
        self.critic_conv = SAGEConv(hidden_size, critic_dom_embed_size)
        self.critic_near_completion = nn.GRU(
            input_size=objective_active_size + objective_indicator_size,
            hidden_size=8,
            num_layers=2,
            # bidirectional=True,
        )
        self.critic_mp_score = nn.Sequential(
            init_xn(nn.Linear(critic_size, objective_indicator_size), "relu"), nn.ReLU()
        )
        self.critic_ap_score = nn.Sequential(
            init_xn(nn.Linear(critic_size, objective_indicator_size), "relu"), nn.ReLU()
        )
        self.graph_size_norm = GraphSizeNorm()
        critic_gate_size = (
            2 * critic_dom_embed_size + 2 * objective_indicator_size + 1 + 1
        )  # active steps, near_completion
        self.critic_gate = nn.Sequential(
            init_xu(nn.Linear(critic_gate_size, critic_gate_size), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(critic_gate_size, 1), "linear"),
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
                    add_self_loops(dom.dom_edge_index)[0],
                )
            )
            x = torch.cat([x, _add_x], dim=1)

        _add_x = self.dom_fn(x)

        x = torch.cat([x, _add_x], dim=1)
        proj_x = self.global_conv(x, full_edges(dom.dom_edge_index))
        self.last_tensors["proj_x"] = proj_x
        full_x = x + proj_x

        # leaves are targetable elements
        leaf_t_mask = leaves.mask.type(torch.float)
        # compute a leaf dependent dom feats to indicate relevent nodes
        leaves_att, leaves_ew = self.leaves_conv(
            safe_bc(proj_x, leaves.dom_index),
            leaf_t_mask,
            add_self_loops(reverse_edges(leaves.edge_index))[0],
        )
        leaves_att = torch.max(leaves_att, leaf_t_mask)
        self.last_tensors["leaves_edge_weights"] = leaves_ew

        # infer action from dom nodes
        full_dom_tag = full_x[:, 0 : self.text_embed_size]
        full_dom_classes = full_x[:, self.text_embed_size : 2 * self.text_embed_size]
        full_dom_text = full_x[:, 2 * self.text_embed_size : 3 * self.text_embed_size]
        dom_ux_action = (
            self.dom_tag(full_dom_tag) * (1 - self.dom_ux_action_decoder_weight[0])
            + self.action_decoder(full_dom_tag) * self.dom_ux_action_decoder_weight[0]
            + self.dom_classes(full_dom_classes)
            * (1 - self.dom_ux_action_decoder_weight[1])
            + self.action_decoder(full_dom_classes)
            * self.dom_ux_action_decoder_weight[1]
            + self.dom_text(full_dom_text) * (1 - self.dom_ux_action_decoder_weight[2])
            + self.action_decoder(full_dom_text) * self.dom_ux_action_decoder_weight[2]
        )
        dom_ux_action = self.dom_ux_action_fn(dom_ux_action)
        # for dom action input == paste
        _dom_ux_action_input = (dom_ux_action[:, 1] + dom_ux_action[:, 3]).view(-1, 1)
        dom_ux_action = torch.cat(
            [
                dom_ux_action[:, 0].view(-1, 1),
                _dom_ux_action_input,
                dom_ux_action[:, 2].view(-1, 1),
                _dom_ux_action_input,
            ],
            dim=1,
        )
        leaf_ux_action = safe_bc(dom_ux_action, leaves.dom_leaf_index)

        # compute an objective dependent dom feats
        # start with word embedding similarities
        obj_sim = lambda tag, obj: F.cosine_similarity(
            safe_bc(dom[tag], objectives_projection.dom_index),
            safe_bc(objectives[obj], objectives_projection.field_index),
            dim=1,
        ).view(-1, 1)
        obj_dom_tag = safe_bc(
            self.objective_dom_tag(dom.tag), objectives_projection.dom_index
        )
        obj_tag_sim = obj_dom_tag * safe_bc(
            objectives.query, objectives_projection.field_index
        )
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
        obj_ux_action_attr = torch.softmax(
            self.objective_ux_action_attr(obj_ux_action).view(
                -1, self.attr_similarity_size, 3
            ),
            dim=2,
        )
        obj_int = obj_ux_action_attr[:, :, 1]
        dom_obj_int = safe_bc(obj_int, objectives_projection.field_index)
        # ux action tag sensitivities
        obj_att_input = (
            (dom_obj_int * obj_tag_similarities).max(dim=1, keepdim=True).values
        )
        if SAFETY:
            assert (
                global_max_pool(obj_att_input, objectives_projection.batch).min().item()
                > 0
            )

        self.last_tensors["obj_att_input"] = obj_att_input
        self.last_tensors["obj_tag_similarities"] = obj_tag_similarities
        _op_proj_x = safe_bc(proj_x, objectives_projection.dom_index)
        obj_att_from_dom, obj_ew = self.objective_conv(
            _op_proj_x,
            obj_att_input,
            add_self_loops(objectives_projection.dom_edge_index)[0],
        )
        obj_att_from_pos, obj_pos_ew = self.objective_pos_conv(
            objectives_projection.edge_attr,
            obj_att_input,
            objectives_projection.edge_index,
        )
        obj_att = torch.max(
            obj_att_from_pos, torch.max(obj_att_from_dom, obj_att_input)
        )

        # project into main action space
        _obj_ux_action = safe_bc(
            obj_ux_action.view(-1, 1), actions.field_ux_action_index
        )

        _leaf_ux_action = safe_bc(
            leaf_ux_action.view(-1, 1), actions.leaf_ux_action_index
        )
        _obj_mask = safe_bc(obj_att, actions.field_dom_index)
        _leaf_mask = safe_bc(leaves_att, actions.leaf_dom_index)

        if SAFETY:
            assert global_max_pool(_obj_mask, actions.batch).min().item() > 0
            assert global_max_pool(_leaf_mask, actions.batch).min().item() > 0

        ux_action_consensus = _leaf_ux_action * _obj_ux_action
        dom_interest = _obj_mask * _leaf_mask

        # begin determining active step
        # compute objective completeness indicators from DOM
        dom_dom_obj_ux_action = safe_bc(dom_ux_action, objectives_projection.dom_index)
        # [0 - 1]
        obj_comp = obj_ux_action_attr[:, :, 2]
        dom_obj_comp = safe_bc(obj_comp, objectives_projection.field_index)
        dom_obj_comp = (
            (dom_obj_comp * obj_tag_similarities).max(dim=1, keepdim=True).values
        )
        dom_obj_ux_action = safe_bc(obj_ux_action, objectives_projection.field_index)
        dom_obj_comp_ux_action = (
            (dom_dom_obj_ux_action * dom_obj_ux_action).max(dim=1, keepdim=True).values
        )

        self.last_tensors["dom_obj_comp"] = dom_obj_comp

        dom_obj_indicator = dom_obj_comp * obj_att * dom_obj_comp_ux_action
        obj_indicator = torch.relu(
            global_max_pool(dom_obj_indicator, objectives_projection.field_index)
        )
        critic_obj_indicator = obj_indicator.detach()

        # determine if a goal has focus
        goal_focus = safe_bc(dom.focused, actions.dom_index)
        goal_focus = global_max_pool(
            goal_focus * ux_action_consensus * dom_interest, actions.field_index
        )
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
        obj_active_seq, _ = self.objective_active(obj_ind_seq)
        obj_active = unpack_sequence(obj_active_seq)[:, -1:]
        obj_active = softmax(obj_active, objectives.batch)
        _obj_active = safe_bc(obj_active, actions.field_index)

        self.last_tensors["obj_indicator"] = obj_indicator
        self.last_tensors["obj_active"] = obj_active
        self.last_tensors["obj_edge_weights"] = obj_ew

        # main action space activity
        action_consensus = torch.relu(ux_action_consensus * dom_interest * _obj_active)

        if SAFETY:
            assert global_max_pool(ux_action_consensus, actions.batch).min().item() > 0
            assert global_max_pool(dom_interest, actions.batch).min().item() > 0
            assert global_max_pool(action_consensus, actions.batch).min().item() > 0

        self.last_tensors["action_consensus"] = action_consensus
        self.last_tensors["ux_action_consensus"] = ux_action_consensus
        self.last_tensors["dom_interest"] = dom_interest
        trunk_max = global_max_pool(action_consensus, actions.action_index)
        # gather the batch ids for the votes
        action_batch_idx = global_max_pool(
            actions.batch.view(-1, 1), actions.action_index
        ).view(-1)
        trunk_softmax = softmax(trunk_max, action_batch_idx)
        trunk_norm = self.trunk_norm(trunk_max, action_batch_idx)
        trunk = torch.cat([trunk_max, trunk_softmax, trunk_norm,], dim=1,)
        trunk = torch.cat([trunk, trunk * trunk_max], dim=1)
        action_votes = self.actor_gate(trunk)
        action_idx = global_max_pool(actions.action_idx, actions.action_index)

        self.last_tensors["obj_att"] = obj_att
        self.last_tensors["leaves_att"] = leaves_att
        self.last_tensors["obj_ux_action"] = obj_ux_action
        self.last_tensors["leaf_ux_action"] = leaf_ux_action

        # critic senses goal completion
        critic_obj_indicator_order = torch.cat(
            [obj_indicator_order.detach(), critic_obj_indicator], dim=1
        )
        critic_obj_ind_seq = pack_as_sequence(
            critic_obj_indicator_order, objectives.batch
        )
        critic_obj_active_seq, critic_near_completion = self.critic_near_completion(
            critic_obj_ind_seq
        )
        critic_obj_active = unpack_sequence(critic_obj_active_seq)[:, -1:]
        # layers * directions, batch, hidden_size
        critic_near_completion = critic_near_completion[-1, :, -1:]

        _critic_obj_active = safe_bc(
            critic_obj_active, objectives_projection.field_index
        )
        critic_active_steps = torch.tanh(
            global_add_pool(critic_obj_active, objectives.batch) - 1
        )

        # critic senses dom
        critic_conv_x = self.critic_conv(x.detach(), dom.dom_edge_index)
        critic_conv_mp = torch.relu(global_max_pool(critic_conv_x, dom.batch))
        critic_conv_ap = torch.relu(
            global_add_pool(self.graph_size_norm(critic_conv_x, dom.batch), dom.batch)
        )

        # critic senses goal difficulty
        critic_dom_ux_action = safe_bc(
            dom_ux_action.detach(), objectives_projection.dom_index
        )
        critic_obj_ux_action = safe_bc(
            obj_ux_action.detach(), objectives_projection.field_index
        )
        critic_obj_active = safe_bc(
            obj_active.detach(), objectives_projection.field_index
        )
        critic_dom_obj_comp_ux_action = dom_obj_comp_ux_action.detach()
        critic_obj_att = obj_att.detach()
        x_critic_input = torch.cat(
            [
                obj_tag_similarities,
                critic_obj_ux_action,
                critic_dom_ux_action,
                critic_obj_att,
                critic_dom_obj_comp_ux_action,
                safe_bc(critic_conv_x, objectives_projection.dom_index),
            ],
            dim=1,
        )

        critic_mp = self.critic_mp_score(x_critic_input) * _critic_obj_active
        critic_ap = self.critic_ap_score(x_critic_input) * _critic_obj_active

        # max/add objective difficulty in batch
        critic_mp = torch.relu(global_max_pool(critic_mp, objectives_projection.batch))
        critic_ap = torch.relu(
            global_add_pool(
                self.graph_size_norm(critic_ap, objectives_projection.field_index),
                objectives_projection.batch,
            )
        )

        critic_x = torch.cat(
            [
                critic_mp,
                critic_ap,
                critic_conv_mp,
                critic_conv_ap,
                critic_active_steps,
                critic_near_completion,
            ],
            dim=1,
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
