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
    GAE as BaseGAE,
)
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import softmax

from a2c_ppo_acktr.model import NNBase, Policy
from a2c_ppo_acktr.utils import init

from .distributions import NodeObjective
from .functions import *
from .nn import EdgeMask, EdgeAttrs
from .factory_resolver import FactoryResolver
from .domx import short_embed, text_embed_size


class GraphPolicy(Policy):
    """
    Wraps Policy class to handle graph inputs via receipts
    """

    def __init__(self, *args, receipts, **kwargs):
        super().__init__(*args, **kwargs)
        self.receipts = receipts
        # patch distributions to handle node based selection
        self.dist = NodeObjective()

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        inputs = self.receipts.redeem(inputs)
        return super().act(inputs, rnn_hxs, masks, deterministic)

    def get_value(self, inputs, rnn_hxs, masks):
        inputs = self.receipts.redeem(inputs)
        return super().get_value(inputs, rnn_hxs, masks)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        inputs = self.receipts.redeem(inputs)
        return super().evaluate_actions(inputs, rnn_hxs, masks, action)


class DirectionalPropagation(nn.Module):
    def __init__(self, hidden_size, transitivity_size, mask_dim, K):
        super().__init__()
        self.transitivity_fn = EdgeAttrs(
            input_dim=hidden_size, out_dim=transitivity_size, prior_edge_size=8
        )
        self.edge_mask = EdgeMask(transitivity_size, mask_dim, K)

    def mask(self, x, dom, projection, mask, resolver):
        dom_edge_trans = self.transitivity_fn(x, dom.edge_index, dom.dom_edge_attr)
        if projection is not dom:
            _dom_edge_trans = safe_bc_edges(dom_edge_trans, projection.br_edge_index)
        else:
            _dom_edge_trans = dom_edge_trans
        dom_mask, dom_mask_ew = self.edge_mask(
            _dom_edge_trans, mask, projection.edge_index,
        )
        resolver["edge_weights"] = dom_mask_ew
        return dom_mask

    def forward(self, x, dom, projection, mask):
        with FactoryResolver(self, x=x, dom=dom, mask=mask, projection=projection) as r:
            return torch.max(mask, r["mask"])


class SoftActionDecoder(nn.Module):
    def __init__(self, num_of_actions):
        super().__init__()
        self.default_action = 1  # input goal
        self.action_words = [
            ["click", "submit", "target", "focus"],
            ["select", "text", "noun", "username", "password"],
            ["copy"],
            ["paste"],
        ][:num_of_actions]
        action_size = len(self.action_words)

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
            init_ot(nn.Linear(action_size, action_size)), nn.Softmax(dim=1)
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
    return torch.cat([dom.text, dom.value, dom.tag, dom.classes], dim=1,)


def domain(target_domain=None):
    def f(func):
        func.__target_domain__ = target_domain
        return func

    return f


class RelaxedBase(NNBase):
    def __init__(
        self,
        input_dim,
        dom_encoder,
        recurrent=False,
        text_embed_size=text_embed_size,
        num_of_actions=4,
    ):
        action_one_hot_size = num_of_actions  # 5 to enable wait
        encoder_size = dom_encoder.out_channels
        hidden_size = (
            3  # radio checked, focus, tampered
            + dom_encoder.out_channels
            + dom_encoder.in_channels
            + 8  # tag sim
            + 2 * text_embed_size  # key, value
            + action_one_hot_size
        )
        super().__init__(recurrent, hidden_size, hidden_size)
        self.hidden_size = hidden_size
        self.dom_encoder = dom_encoder

        self.objective_active = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
        )
        # self.active_fn = nn.Sequential(init_xn(nn.Linear(hidden_size * 2, 1), "linear"))
        critic_embed_size = 16
        self.critic_mp_score = nn.Sequential(
            init_xu(nn.Linear(hidden_size, critic_embed_size), "relu"), nn.ReLU()
        )
        self.critic_ap_score = nn.Sequential(
            init_xu(nn.Linear(hidden_size, critic_embed_size), "relu"), nn.ReLU()
        )
        critic_gate_size = (
            2 * critic_embed_size + 4 * hidden_size
        )  # active steps, near_completion
        self.critic_gate = nn.Sequential(
            init_xu(nn.Linear(critic_gate_size, hidden_size), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(hidden_size, 1), "linear"),
        )

    @domain("dom")
    def encoder_input(self, dom):
        return autoencoder_x(dom)

    @domain("dom")
    def x(self, dom, encoded_x, encoder_input):
        return torch.cat(
            [encoder_input, encoded_x, dom.radio_value, dom.focused, dom.tampered],
            dim=1,
        )

    @domain("action")
    def action_x(self, action, field, x, obj_tag_similarities):
        _x = safe_bc(x, action.dom_index)
        _fk = safe_bc(torch.cat([field.key, field.query], dim=1), action.field_index)
        _ts = safe_bc(obj_tag_similarities, action.field_dom_index)
        return torch.cat([_x, _fk, action.action_one_hot, _ts], dim=1)

    @domain("dom")
    def encoded_x(self, dom, encoder_input):
        with torch.no_grad():
            return torch.tanh(self.dom_encoder(encoder_input, dom.edge_index))

    @domain("dom_field")
    def obj_tag_similarities(self, dom, field, dom_field):
        # compute an objective dependent dom feats
        # start with word embedding similarities
        obj_sim = lambda tag, obj: F.cosine_similarity(
            safe_bc(dom[tag], dom_field.dom_index),
            safe_bc(field[obj], dom_field.field_index),
            dim=1,
        ).view(-1, 1)
        return torch.relu(
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

    def action_batch_idx(self, action):
        return global_max_pool(action.batch.view(-1, 1), action.action_index).view(-1)

    @domain("field")
    def obj_active(self, field, action, z, edge_weights, resolver):
        raise NotImplementedError

    def action_votes(self, action, z, obj_active, edge_weights):
        raise NotImplementedError

    def forward(self, inputs, rnn_hxs, masks):
        from torch_geometric.data import Batch, Data

        assert isinstance(inputs, tuple), str(type(inputs))
        assert all(map(lambda x: isinstance(x, Batch), inputs))
        dom, objectives, objectives_projection, leaves, actions, *history = inputs
        assert actions.edge_index is not None, str(inputs)

        with FactoryResolver(
            self,
            dom=dom,
            field=objectives,
            dom_leaf=leaves,
            dom_field=objectives_projection,
            action=actions,
            rnn_hxs=rnn_hxs,
            masks=masks,
        ) as r:
            return (
                r["critic_value"],
                (r["action_votes"], r["action_batch_idx"]),
                r["rnn_hxs"],
            )

    def critic_value(self, critic_x):
        return self.critic_gate(critic_x)

    def critic_x(self, field, action, action_x, obj_active, obj_active_mem):
        # layers * directions, batch, hidden_size
        critic_near_completion = obj_active_mem.view(obj_active_mem.shape[1], -1)

        critic_mp = self.critic_mp_score(action_x)
        critic_ap = self.critic_ap_score(action_x)

        # max/add objective difficulty in batch
        critic_mp = torch.relu(global_max_pool(critic_mp, action.batch))
        critic_ap = torch.tanh(global_add_pool(critic_ap, action.batch))

        critic_x = torch.cat([critic_mp, critic_ap, critic_near_completion], dim=1)
        return critic_x


class SAGEConvBase(RelaxedBase):
    def __init__(
        self,
        input_dim,
        dom_encoder,
        recurrent=False,
        text_embed_size=text_embed_size,
        num_of_actions=4,
    ):
        super().__init__(
            input_dim, dom_encoder, recurrent, text_embed_size, num_of_actions
        )

        self.global_conv = SAGEConv(self.hidden_size, self.hidden_size)
        self.active_conv = SAGEConv(self.hidden_size, self.hidden_size)
        self.vote_conv = SAGEConv(self.hidden_size * 3, self.hidden_size)
        self.vote_fn = nn.Sequential(
            init_xu(nn.Linear(self.hidden_size, self.hidden_size // 2), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(self.hidden_size // 2, 1), "relu"),
            nn.ReLU(),
        )

    @domain("action")
    def z(self, action, action_x):
        return torch.relu(self.global_conv(action_x, action.edge_index))

    @domain("field")
    def obj_active(self, field, action, z, resolver):
        active = self.active_conv(z, action.edge_index)
        _active = global_max_pool(active, action.field_index)
        obj_ind_seq = pack_as_sequence(_active, field.batch)
        obj_active_seq, obj_active_mem = self.objective_active(obj_ind_seq)
        resolver["obj_active_mem"] = obj_active_mem
        return unpack_sequence(obj_active_seq)

    def action_votes(self, action, z, obj_active):
        z1 = torch.cat([z, safe_bc(obj_active, action.field_index)], dim=1)
        z2 = self.vote_conv(z1, action.edge_index)
        z3 = global_max_pool(z2, action.action_index)
        return self.vote_fn(z3)


class GCNConvBase(RelaxedBase):
    def __init__(
        self,
        input_dim,
        dom_encoder,
        recurrent=False,
        text_embed_size=text_embed_size,
        num_of_actions=4,
    ):
        super().__init__(
            input_dim, dom_encoder, recurrent, text_embed_size, num_of_actions
        )
        self.edge_weights_fn = nn.Sequential(
            init_xn(nn.Linear(8, 1), "sigmoid"), nn.Sigmoid()
        )
        self.global_conv = GCNConv(self.hidden_size, self.hidden_size)
        self.active_conv = GCNConv(self.hidden_size, self.hidden_size)
        self.vote_conv = GCNConv(self.hidden_size, 1)

    @domain("action")
    def edge_weights(self, dom, action):
        return safe_bc(self.edge_weights_fn(dom.edge_attr), action.br_edge_index).view(
            -1
        )

    @domain("action")
    def z(self, action, action_x, edge_weights):
        return torch.relu(self.global_conv(action_x, action.edge_index, edge_weights))

    @domain("field")
    def obj_active(self, field, action, z, edge_weights, resolver):
        active = self.active_conv(z, action.edge_index, edge_weights)
        _active = global_max_pool(active, action.field_index)
        obj_ind_seq = pack_as_sequence(_active, field.batch)
        obj_active_seq, obj_active_mem = self.objective_active(obj_ind_seq)
        resolver["obj_active_mem"] = obj_active_mem
        # [0, 1]
        return softmax(self.active_fn(unpack_sequence(obj_active_seq)), field.batch)

    def action_votes(self, action, z, obj_active, edge_weights):
        z2 = self.vote_conv(z, action.edge_index, edge_weights)
        return torch.relu(
            global_max_pool(
                z2 * safe_bc(obj_active, action.field_index), action.action_index
            )
        )


class GNNBase(NNBase):
    """
    DOM Based Actor / Restricted model
    """

    def __init__(
        self,
        input_dim,
        dom_encoder,
        recurrent=False,
        text_embed_size=text_embed_size,
        num_of_actions=4,
    ):
        action_one_hot_size = num_of_actions  # 5 to enable wait
        encoder_size = dom_encoder.out_channels
        # radio checked, focus, tampered
        x_size = hidden_size = 3 + dom_encoder.out_channels + dom_encoder.in_channels
        super().__init__(recurrent, hidden_size, hidden_size)
        self.hidden_size = hidden_size
        self.text_embed_size = text_embed_size

        self.dom_encoder = dom_encoder
        self.attr_norm = nn.BatchNorm1d(text_embed_size)
        self.action_decoder = SoftActionDecoder(num_of_actions)

        # paste  & copy have equal weight
        if num_of_actions > 3:
            dom_ux_action_fn_size = action_one_hot_size - 1
        else:
            dom_ux_action_fn_size = action_one_hot_size
        self.dom_ux_action_fn = nn.Sequential(
            init_xn(nn.Linear(encoder_size, dom_ux_action_fn_size), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(dom_ux_action_fn_size, dom_ux_action_fn_size), "linear"),
            nn.Softmax(dim=1),
        )
        transitivity_size = 12
        self.leaf_prop = DirectionalPropagation(
            encoder_size, transitivity_size, mask_dim=1, K=5
        )
        self.goal_prop = self.leaf_prop
        self.value_prop = self.leaf_prop
        """
        self.goal_prop = DirectionalPropagation(
            encoder_size, transitivity_size, mask_dim=1, K=5
        )
        self.value_prop = DirectionalPropagation(
            encoder_size, transitivity_size, mask_dim=1, K=5
        )
        """

        self.dom_description_fn = nn.Sequential(
            init_xn(nn.Linear(encoder_size, text_embed_size), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(text_embed_size, text_embed_size), "tanh"),
            nn.Tanh(),
        )
        # attr similarities
        self.attr_similarity_size = attr_similarity_size = 10
        # [ goal att, value ]
        self.dom_objective_attr_fn = nn.Sequential(
            init_xn(nn.Linear(encoder_size, 2 * attr_similarity_size), "relu"),
            nn.ReLU(),
            init_xn(
                nn.Linear(2 * attr_similarity_size, 2 * attr_similarity_size), "sigmoid"
            ),
            nn.Sigmoid(),
        )
        # [ enabled ]
        self.dom_disabled_fn = nn.Sequential(
            init_xn(nn.Linear(x_size, 8), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(8, 1), "sigmoid"),
            nn.Sigmoid(),
        )
        self.dom_disabled_prop = self.leaf_prop
        # self.dom_disabled_prop = DirectionalPropagation(
        #    encoder_size, transitivity_size, mask_dim=1, K=5
        # )

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
            init_xn(nn.Linear(action_one_hot_size, 1), "sigmoid"), nn.Sigmoid()
        )
        self.dom_needs_value_fn = nn.Sequential(
            init_xn(nn.Linear(x_size, 8), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(8, 1), "sigmoid"),
            nn.Sigmoid(),
        )
        self.dom_disabled_value_fn = nn.Sequential(
            init_xn(nn.Linear(x_size, 8), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(8, 1), "sigmoid"),
            nn.Sigmoid(),
        )
        self.dom_disabled_value_prop = DirectionalPropagation(
            encoder_size, transitivity_size, mask_dim=1, K=5
        )
        # has_value, order, is last, has focus, ux action, tampered
        objective_active_size = 5 + action_one_hot_size
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
            num_layers=1,
            bidirectional=True,
        )
        self.active_fn = nn.Sequential(
            init_xn(nn.Linear(objective_active_size * 2, 1), "linear")
        )
        self.trunk_norm = InstanceNorm(1)
        # ac_norm, action_status_indicators, active_mem
        trunk_size = 3  # 2 * objective_active_size * 2
        self.actor_gate = nn.Sequential(
            init_ones(nn.Linear(trunk_size, 1), "relu"), nn.ReLU()
        )
        self.critic_size = critic_size = (
            14
            + 2 * action_one_hot_size
            + objective_active_size
            + 3 * attr_similarity_size
            + 2 * objective_active_size
        )
        self.critic_mp_score = nn.Sequential(
            init_xn(nn.Linear(critic_size, critic_size), "relu"),
            nn.ReLU(),
            init_xu(nn.Linear(critic_size, critic_size), "relu"),
            nn.ReLU(),
        )
        self.critic_ap_score = nn.Sequential(
            init_xn(nn.Linear(critic_size, critic_size), "relu"),
            nn.ReLU(),
            init_xu(nn.Linear(critic_size, critic_size), "relu"),
            nn.ReLU(),
        )
        critic_gate_size = 2 * critic_size
        self.critic_gate = nn.Sequential(
            init_xn(nn.Linear(critic_gate_size, 1), "linear"),
        )

    @domain("dom")
    def encoder_input(self, dom):
        return autoencoder_x(dom)

    @domain("dom")
    def x(self, dom, encoded_x, encoder_input):
        return torch.cat(
            [encoder_input, encoded_x, dom.radio_value, dom.focused, dom.tampered],
            dim=1,
        )

    @domain("dom")
    def encoded_x(self, dom, encoder_input):
        with torch.no_grad():
            return torch.tanh(self.dom_encoder(encoder_input, dom.edge_index))

    @domain("dom_leaf")
    def leaves_att(self, dom, dom_leaf, encoded_x):
        leaf_t_mask = dom_leaf.mask.type(torch.float)
        return self.leaf_prop(encoded_x, dom, dom_leaf, leaf_t_mask)

    @domain("dom")
    def dom_ux_action(self, encoded_x):
        # infer action from dom nodes
        dom_ux_action = self.dom_ux_action_fn(encoded_x)
        # for dom action input == paste
        if dom_ux_action.shape[1] > 2:
            return torch.cat(
                [dom_ux_action[:, 0:], dom_ux_action[:, 1].view(-1, 1)], dim=1
            )
        return dom_ux_action

    @domain("dom")
    def dom_description(self, encoded_x):
        return self.dom_description_fn(encoded_x)

    @domain("dom_leaf")
    def leaf_ux_action(self, dom_ux_action, dom_leaf):
        return global_max_pool(
            safe_bc(dom_ux_action, dom_leaf.dom_index) * dom_leaf.mask,
            dom_leaf.leaf_index,
        )

    @domain("dom_field")
    def obj_tag_similarities(self, dom, field, dom_field, dom_description):
        # compute an objective dependent dom feats
        # start with word embedding similarities
        obj_sim = lambda tag, obj: F.cosine_similarity(
            safe_bc(dom[tag], dom_field.dom_index),
            safe_bc(field[obj], dom_field.field_index),
            dim=1,
        ).view(-1, 1)
        _dom_desc = safe_bc(dom_description, dom_field.dom_index)
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
    def dom_obj_attr(self, dom_field, encoded_x):
        """
        Compute attribute affinity to goal target or completion
        Returns: [:,:,0] = goal target. [:,:,1] = goal completion.
        [0,1]
        """
        dom_obj_attr = self.dom_objective_attr_fn(encoded_x).view(
            encoded_x.shape[0], self.attr_similarity_size, 2
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
    def obj_att_input(self, dom, dom_field, encoded_x, dom_obj_int):
        obj_att_input = torch.relu(
            (dom_obj_int[:, :, 0]).max(dim=1, keepdim=True).values
        )

        if SAFETY:
            assert global_max_pool(obj_att_input, dom_field.batch).min().item() > 0

        return self.goal_prop(encoded_x, dom, dom_field, obj_att_input)

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
        return (
            ux_action_consensus.max(dim=1, keepdim=True).values
            * disabled_mask
            * obj_int
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
    def dom_disabled_value_mask(self, dom, x, encoded_x):
        dom_disabled = self.dom_disabled_value_fn(x)
        # invert after propagation
        return 1 - self.dom_disabled_value_prop(encoded_x, dom, dom, dom_disabled)

    @domain("dom")
    def dom_needs_value(self, x):
        return self.dom_needs_value_fn(x)

    @domain("dom_field")
    def value_mask(
        self,
        dom,
        dom_field,
        encoded_x,
        x,
        dom_obj_int,
        dom_needs_value,
        dom_disabled_value_mask,
    ):
        """
        compute objective completeness with goal word similarities
        [0 - 1]
        """
        obj_value_mask = (dom_obj_int[:, :, 1]).max(dim=1, keepdim=True).values
        _dom_value_mask = safe_bc(dom_disabled_value_mask, dom_field.dom_index)
        _non_value = safe_bc(dom_needs_value, dom_field.dom_index)
        value_mask = (
            torch.cat([obj_value_mask, _non_value], dim=1)
            .max(dim=1, keepdim=True)
            .values
        )
        value_mask = (
            self.value_prop(encoded_x, dom, dom_field, value_mask) * _dom_value_mask
        )
        return value_mask

    @domain("dom")
    def disabled_mask(self, dom, encoded_x, x):
        """
        determine if a node is active/interesting
        """
        mask = self.dom_disabled_fn(x)
        # invert after positive propagation
        return 1 - self.dom_disabled_prop(encoded_x, dom, dom, mask)

    @domain("field")
    def obj_needs_value(self, obj_ux_action):
        return self.ux_action_needs_value_fn(obj_ux_action)

    @domain("action")
    def action_status_indicators(
        self, dom, action, dom_interest_norm, value_mask, obj_needs_value
    ):
        """
        determine if a node supplies a goal value (or doesn't need to)
        """
        _value_mask = safe_bc(value_mask, action.field_dom_index)
        _non_value = safe_bc(obj_needs_value, action.field_index)
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
        encoded_x,
        x,
        rnn_hxs,
        masks,
        resolver,
    ):
        """
        Input for determining the active step
        """
        # has the value been supplied?
        status_indicators = global_max_pool(
            action_status_indicators, action.field_index
        )

        dom_obj_focus = safe_bc(dom.focused, action.dom_index)
        dom_obj_tampered = safe_bc(dom.tampered, action.dom_index)

        # pack on order embed and goal focus
        obj_indicator_order = torch.cat(
            [
                status_indicators,
                field.order,
                field.is_last,
                global_max_pool(dom_obj_focus * dom_interest_norm, action.field_index),
                global_max_pool(
                    dom_obj_tampered * dom_interest_norm, action.field_index
                ),
                obj_ux_action,
            ],
            dim=1,
        )

        if self.is_recurrent:
            goal_dom_input = torch.cat(
                [
                    safe_bc(x, dom_field.dom_index),
                    safe_bc(obj_indicator_order, dom_field.field_index),
                ],
                dim=1,
            )
            goal_dom_encoded = self.goal_dom_encoder(
                goal_dom_input, dom_field.edge_index
            )
            goal_dom_flat = global_max_pool(goal_dom_encoded, dom_field.batch)
            curr_state, rnn_hxs = self._forward_gru(goal_dom_flat, rnn_hxs, masks)
            resolver["rnn_hxs"] = rnn_hxs
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
    def obj_active(self, field, obj_indicator_order, resolver):
        obj_ind_seq = pack_as_sequence(obj_indicator_order, field.batch)
        obj_active_seq, obj_active_mem = self.objective_active(obj_ind_seq)
        resolver["obj_active_mem"] = obj_active_mem
        # [0, 1]
        return softmax(self.active_fn(unpack_sequence(obj_active_seq)), field.batch)

    @domain("action_index")
    def action_trunk(self, action, ac_norm, action_status_indicators, obj_active):
        _obj_active = safe_bc(obj_active, action.field_index)
        trunk = torch.cat(
            [ac_norm, ac_norm - action_status_indicators, _obj_active * ac_norm], dim=1
        )
        return global_max_pool(trunk, action.action_index)

    def action_votes(self, action_trunk):
        return self.actor_gate(action_trunk)

    def action_batch_idx(self, action):
        return global_max_pool(action.batch.view(-1, 1), action.action_index).view(-1)

    def forward(self, inputs, rnn_hxs, masks):
        from torch_geometric.data import Batch, Data

        assert isinstance(inputs, tuple), str(type(inputs))
        assert all(map(lambda x: isinstance(x, Batch), inputs))
        dom, objectives, objectives_projection, leaves, actions, *history = inputs
        assert actions.edge_index is not None, str(inputs)

        with FactoryResolver(
            self,
            dom=dom,
            field=objectives,
            dom_leaf=leaves,
            dom_field=objectives_projection,
            action=actions,
            rnn_hxs=rnn_hxs,
            masks=masks,
        ) as r:
            return (
                r["critic_value"],
                (r["action_votes"], r["action_batch_idx"]),
                r["rnn_hxs"],
            )

    @domain("action")
    def critic_trunk(
        self,
        action,
        ac_value,
        ac_norm,
        action_status_indicators,
        obj_active,
        value_mask,
        disabled_mask,
        obj_ux_action,
        dom_interest,
        action_consensus,
        ux_action_consensus,
        leaves_att,
        dom_ux_action,
        leaf_ux_action,
        obj_tag_similarities,
        dom_obj_int,
        obj_att_input,
        dom_disabled_value_mask,
        dom_needs_value,
        obj_needs_value,
        obj_indicator_order,
        obj_active_mem,
    ):
        _obj_active = safe_bc(obj_active, action.field_index)
        _value_mask = safe_bc(value_mask, action.field_dom_index)
        _disabled_mask = safe_bc(disabled_mask, action.dom_index)
        _dom_needs_value = safe_bc(dom_needs_value, action.dom_index)
        _obj_needs_value = safe_bc(obj_needs_value, action.field_index)
        _leaves_att = safe_bc(leaves_att, action.dom_leaf_index)
        # action size
        _dom_ux_action = safe_bc(dom_ux_action, action.dom_index)
        _leaf_ux_action = safe_bc(leaf_ux_action, action.leaf_index)
        # tag sim size
        _obj_tag_similarities = safe_bc(obj_tag_similarities, action.field_dom_index)
        _dom_obj_int = safe_bc(dom_obj_int, action.field_dom_index)
        _obj_att_input = safe_bc(obj_att_input, action.field_dom_index)
        _dom_disabled_value_mask = safe_bc(dom_disabled_value_mask, action.dom_index)
        _obj_indicator_order = safe_bc(obj_indicator_order, action.field_index)
        _obj_active_mem = safe_bc(
            obj_active_mem.view(obj_active_mem.shape[1], -1), action.action_index
        )
        _ac_value = safe_bc(ac_value, action.action_index)

        trunk = torch.cat(
            [
                action_status_indicators,
                _obj_active,
                ac_norm,
                _ac_value,
                _value_mask,
                _disabled_mask,
                _dom_needs_value,
                _obj_needs_value,
                _leaves_att,
                _dom_obj_int.view(_dom_obj_int.shape[0], -1),
                _obj_att_input,
                _dom_disabled_value_mask,
                dom_interest,
                action_consensus,
                ux_action_consensus,
                _dom_ux_action,
                _leaf_ux_action,
                _obj_indicator_order,
                _obj_tag_similarities,
                _obj_active_mem,
            ],
            dim=1,
        )
        return trunk

    def critic_value(self, critic_x):
        return self.critic_gate(critic_x)

    def critic_x(self, field, action, critic_trunk):
        critic_mp = self.critic_mp_score(critic_trunk)
        critic_ap = self.critic_ap_score(critic_trunk)

        # max/add objective difficulty in batch
        critic_mp = torch.relu(global_max_pool(critic_mp, action.batch))
        critic_ap = torch.tanh(global_add_pool(critic_ap, action.batch))

        critic_x = torch.cat([critic_mp, critic_ap], dim=1)
        return critic_x


class Instructor(GNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instructor_size = self.hidden_size
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
            list(map(lambda x: torch.from_numpy(short_embed(x)), self.target_words))
        ).view(1, -1)
        self.key_selection_size = len(self.target_words) + 4
        self.key_softmax_fn = nn.Sequential(
            init_xn(nn.Linear(self.instructor_size, self.text_embed_size), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(self.text_embed_size, self.key_selection_size)),
        )
        self.query_softmax_fn = nn.Sequential(
            init_xn(nn.Linear(self.instructor_size, self.text_embed_size), "relu"),
            nn.ReLU(),
            init_xn(nn.Linear(self.text_embed_size, 4)),
        )
        self.norm_query = InstanceNorm(1)

    def field(self, dom, _field, x):
        batch_size = _field.batch.shape[0]
        device = x.device
        key_softmax_x = global_max_pool(self.key_softmax_fn(x), dom.batch)
        key_softmax = F.softmax(key_softmax_x, dim=1).repeat_interleave(
            self.text_embed_size, dim=1
        )
        key = self.key_fn(x, dom.batch)[:, : self.text_embed_size * 4]
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
        query_softmax_x = global_max_pool(self.query_softmax_fn(x), dom.batch)
        query_softmax = F.softmax(query_softmax_x, dim=1).repeat_interleave(
            self.text_embed_size, dim=1
        )
        query = self.query_fn(x, dom.batch)[:, : self.text_embed_size * 4]
        field_query = (
            (query * query_softmax)
            .view(batch_size, self.text_embed_size, 4)
            .sum(dim=2)
            .view(batch_size * self.text_embed_size, 1)
        )

        _field.query = self.norm_query(
            field_query  # , torch.arange(0, field_query.shape[0], dtype=torch.long)
        ).view(batch_size, self.text_embed_size)
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

        with FactoryResolver(
            self,
            dom=dom,
            _field=objectives,
            dom_leaf=leaves,
            dom_field=objectives_projection,
            action=actions,
            rnn_hxs=rnn_hxs,
            masks=masks,
        ) as r:
            return (
                r["critic_value"],
                (r["action_votes"], r["action_batch_idx"]),
                r["rnn_hxs"],
            )


class GAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_encoder = Encoder(in_channels, out_channels)
        self.ae = BaseGAE(self.edge_encoder)

    def encode(self, x, edge_index):
        return self.ae.encode(x, edge_index)

    def decode(self, z, edge_index):
        return self.ae_decode(z, edge_index)

    def recon_loss(self, z, edge_index):
        return self.ae.recon_loss(z, edge_index)

    def forward(self, z, edge_index):
        return self.encode(z, edge_index)


class DualGAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_index_1_encoder = Encoder(in_channels, out_channels // 2)
        self.edge_index_2_encoder = Encoder(in_channels, out_channels // 2)
        self.ae_1 = GAE(self.edge_index_1_encoder)
        self.ae_2 = GAE(self.edge_index_2_encoder)

    def encode(self, x, edge_index_1, edge_index_2):
        return torch.cat(
            [self.ae_1.encode(x, edge_index_1), self.ae_2.encode(x, edge_index_2),],
            dim=1,
        )

    def decode(self, z, edge_index_1, edge_index_2):
        z1, z2 = z[:, : self.out_channels // 2], z[:, self.out_channels // 2 :]
        return torch.cat(
            [self.ae_1.decode(z1, edge_index_1), self.ae_2.decode(z2, edge_index_2)],
            dim=1,
        )

    def recon_loss(self, z, edge_index_1, edge_index_2):
        return self.ae_1.recon_loss(z, edge_index_1) + self.ae_2.recon_loss(
            z, edge_index_2
        )

    def forward(self, z, edge_index_1, edge_index_2):
        return self.encode(z, edge_index_1, edge_index_2)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, add_self_loops=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GCNConv(
            in_channels, 2 * out_channels,  # add_self_loops=add_self_loops
        )
        self.conv2 = GCNConv(
            2 * out_channels, out_channels,  # add_self_loops=add_self_loops
        )

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


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
