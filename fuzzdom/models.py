import torch
from torch import nn
from torch_geometric.nn import (
    SAGEConv,
    SGConv,
    APPNP,
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
import math

from a2c_ppo_acktr.model import NNBase
from a2c_ppo_acktr.utils import init


init_t = lambda m: init(
    m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), 2 ** 0.5
)
init_r = lambda m: init(
    m,
    nn.init.orthogonal_,
    lambda x: nn.init.constant_(x, 0),
    nn.init.calculate_gain("relu"),
)
init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
init_i = lambda m: init(
    m,
    lambda x, gain: nn.init.ones_(x),
    lambda x: nn.init.constant_(x, 0) if x else None,
)
init_xu = lambda m, nl="relu": init(
    m,
    nn.init.xavier_uniform_,
    lambda x: nn.init.constant_(x, 0),
    nn.init.calculate_gain(nl),
)

reverse_edges = lambda e: torch.stack([e[1], e[0]]).contiguous()
full_edges = lambda e: torch.cat(
    [add_self_loops(e)[0], reverse_edges(e)], dim=1
).contiguous()


class AdditiveMask(nn.Module):
    def __init__(self, input_dim, mask_dim=1, K=5):
        super(AdditiveMask, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.x_fn = nn.Sequential(init_t(nn.Linear(input_dim, input_dim)), nn.Tanh(),)
        self.conv = APPNP(K=K, alpha=self.alpha)
        self.K = K

    def forward(self, x, mask, edge_index):
        x = self.x_fn(x)
        edge_weights = torch.relu(
            F.cosine_similarity(x[edge_index[0]], x[edge_index[1]])
        ).view(-1)
        fill = torch.relu(mask)
        fill = self.conv(fill, edge_index, edge_weights)
        return fill, edge_weights


class GNNBase(NNBase):
    def __init__(
        self,
        input_dim,
        dom_encoder=None,
        recurrent=False,
        hidden_size=64,
        text_embed_size=25,
    ):
        super(GNNBase, self).__init__(
            recurrent, hidden_size, hidden_size,
        )
        self.hidden_size = hidden_size

        self.dom_encoder = dom_encoder

        dom_attr_size = 5
        action_one_hot_size = 5
        query_input_dim = 6 + 2 * dom_attr_size
        if dom_encoder:
            query_input_dim += dom_encoder.out_channels
        self.attr_norm = nn.BatchNorm1d(text_embed_size)
        self.dom_tag = nn.Sequential(
            init_xu(nn.Linear(text_embed_size, dom_attr_size)), nn.ReLU(),
        )
        self.dom_classes = nn.Sequential(
            init_xu(nn.Linear(text_embed_size, dom_attr_size)), nn.ReLU(),
        )
        self.dom_fn = nn.Sequential(
            init_xu(nn.Linear(query_input_dim, hidden_size - query_input_dim), "tanh"),
            nn.Tanh(),
        )
        # self.global_conv = SAGEConv(hidden_size, hidden_size)
        self.global_conv = SGConv(hidden_size, hidden_size, K=7)

        self.dom_ux_action_fn = nn.Sequential(
            init_xu(nn.Linear(hidden_size, action_one_hot_size)),
            nn.ReLU(),
            init_(nn.Linear(action_one_hot_size, action_one_hot_size)),
            nn.Sigmoid(),  # nn.Softmax(dim=1),
        )
        self.leaves_conv = AdditiveMask(hidden_size, 1, K=7)
        # attr similarities
        attr_similarity_size = 8
        self.objective_conv = AdditiveMask(hidden_size, 1, K=7)
        self.objective_int_fn = nn.Sequential(
            init_xu(nn.Linear(attr_similarity_size, 1)), nn.Sigmoid(),
        )

        self.objective_target_embeds = nn.Parameter(
            torch.rand(action_one_hot_size, text_embed_size)
        )
        self.objective_ux_action_fn = nn.Sequential(
            # init_r(nn.Linear(text_embed_size * 2, text_embed_size)),
            # nn.ReLU(),
            init_xu(nn.Linear(text_embed_size, action_one_hot_size)),
            nn.Softmax(dim=1),
            # nn.Linear(action_one_hot_size, action_one_hot_size),
            # nn.Softmax(dim=1),
        )
        # ux similarity * dom intereset, order
        trunk_size = 2
        self.actor_gate = nn.Sequential(init_i(nn.Linear(trunk_size, 1)), nn.ReLU())

        critic_size = hidden_size + action_one_hot_size + attr_similarity_size
        self.critic_obj_fn = nn.Sequential(
            init_xu(nn.Linear(text_embed_size * 2, text_embed_size)),
            nn.ReLU(),
            init_xu(nn.Linear(text_embed_size, action_one_hot_size)),
            nn.ReLU(),
        )
        self.critic_linear_add = nn.Sequential(
            init_xu(nn.Linear(critic_size, hidden_size)), nn.ReLU()
        )
        self.critic_linear_max = nn.Sequential(
            init_xu(nn.Linear(critic_size, hidden_size)), nn.ReLU()
        )
        self.critic_ap_norm = nn.BatchNorm1d(hidden_size)
        self.critic_gate = nn.Sequential(init_xu(nn.Linear(hidden_size * 2, 1)),)
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        from torch_geometric.data import Batch, Data

        assert isinstance(inputs, tuple), str(type(inputs))
        assert all(map(lambda x: isinstance(x, Batch), inputs))
        dom, objectives, objectives_projection, leaves, actions, history = inputs
        assert actions.edge_index is not None, str(inputs)
        self.last_tensors = {}

        x = torch.cat(
            [
                self.dom_tag(dom.tag),
                self.dom_classes(dom.classes),
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
            _add_x = self.dom_encoder(
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
                add_self_loops(dom.edge_index)[0],
            )
            x = torch.cat([x, _add_x], dim=1)

        _add_x = self.dom_fn(x)

        x = torch.cat([x, _add_x], dim=1)
        proj_x = self.global_conv(x, full_edges(dom.edge_index))
        self.last_tensors["proj_x"] = proj_x

        # leaves are targetable elements
        leaf_t_mask = leaves.mask.type(torch.float)
        # compute a leaf dependent dom feats to indicate relevent nodes
        leaves_att, leaves_ew = self.leaves_conv(
            proj_x[leaves.dom_index],
            leaf_t_mask,
            add_self_loops(reverse_edges(leaves.edge_index))[0],
        )
        leaves_att = leaves_att + leaf_t_mask
        self.last_tensors["leaves_edge_weights"] = leaves_ew

        # infer action from dom nodes, but for leaves only
        leaf_ux_action = self.dom_ux_action_fn(x[leaves.dom_index])

        # compute an objective dependent dom feats
        # start with word embedding similarities
        obj_sim = lambda tag, obj: F.cosine_similarity(
            dom[tag][objectives_projection.dom_index],
            objectives[obj][objectives_projection.field_index],
        ).view(-1, 1)
        obj_tag_similarities = torch.cat(
            [
                obj_sim("text", "key"),
                obj_sim("text", "query"),
                obj_sim("value", "key"),
                obj_sim("value", "query"),
                obj_sim("tag", "key"),
                obj_sim("tag", "query"),
                obj_sim("classes", "key"),
                obj_sim("classes", "query"),
            ],
            dim=1,
        )
        obj_att_input = obj_tag_similarities.max(dim=1, keepdim=True)[
            0
        ]  # self.objective_int_fn(obj_tag_similarities)
        self.last_tensors["obj_att_input"] = obj_att_input
        self.last_tensors["obj_tag_similarities"] = obj_tag_similarities
        obj_att, obj_ew = self.objective_conv(
            proj_x[objectives_projection.dom_index],
            obj_att_input,
            add_self_loops(objectives_projection.edge_index)[0],
        )
        obj_att = obj_att + obj_att_input

        self.last_tensors["obj_edge_weights"] = obj_ew
        # infer action from query key

        # obj_ux_action = self.objective_ux_action_fn(objectives.key)

        bs = objectives.key.shape[0]
        # alt method
        """
        sims = F.cosine_similarity(
            objectives.key.repeat(1, 5).view(bs * 5, -1),
            self.objective_target_embeds.repeat(bs, 1),
        ).view(bs, 5)
        obj_ux_action = self.objective_ux_action_fn(sims)
        """
        # debug: set to click
        obj_ux_action = torch.zeros(bs, 5, device="cuda:0")
        obj_ux_action[:, 0] = 1.0
        leaf_ux_action = torch.zeros(leaves.batch.shape[0], 5, device="cuda:0")
        leaf_ux_action[:, 0] = 1.0

        # project into main trunk
        _obj_ux_action = obj_ux_action.view(-1, 1)[actions.field_ux_action_index]
        _leaf_ux_action = leaf_ux_action.view(-1, 1)[actions.leaf_ux_action_index]
        _obj_mask = obj_att[actions.field_index]
        _leaf_mask = leaves_att[actions.dom_leaf_index]

        torch.set_printoptions(profile="full")
        print("action_idx", actions.action_idx.shape)
        print(actions.action_idx)
        print("field_action_index", actions.field_ux_action_index.shape)
        print(actions.field_ux_action_index)
        print("p_action", _obj_ux_action.shape)
        # print(_obj_ux_action)
        print(actions.action_idx > 0)
        print((actions.action_idx > 0) & (_obj_ux_action > 0).squeeze(1))

        assert _obj_ux_action[actions.action_idx > 0].sum().item() == 0, (
            _obj_ux_action[actions.action_idx > 0].sum().item()
        )
        assert _leaf_ux_action[actions.action_idx > 0].sum().item() == 0

        ux_action_consensus = _leaf_ux_action * _obj_ux_action
        dom_interest = _obj_mask * _leaf_mask
        action_consensus = torch.relu(ux_action_consensus * dom_interest) ** 0.5
        self.last_tensors["action_consensus"] = action_consensus
        self.last_tensors["ux_action_consensus"] = ux_action_consensus
        self.last_tensors["dom_interest"] = dom_interest
        trunk = global_max_pool(
            torch.cat(
                [
                    action_consensus,
                    action_consensus * (1 - objectives.order[actions.field_index]),
                ],
                dim=1,
            ),
            actions.action_index,
        )
        action_votes = self.actor_gate(trunk)
        action_batch_idx = global_max_pool(
            actions.batch.view(-1, 1), actions.action_index
        )

        self.last_tensors["obj_att"] = obj_att
        self.last_tensors["leaves_att"] = leaves_att
        self.last_tensors["obj_ux_action"] = obj_ux_action
        self.last_tensors["leaf_ux_action"] = leaf_ux_action

        # critic is aware of objectives & dom
        critic_obj = self.critic_obj_fn(
            torch.cat([objectives.key, objectives.query], dim=1,)
        )
        x_critic_input = torch.cat(
            [
                proj_x[objectives_projection.dom_index].detach(),
                obj_tag_similarities,
                critic_obj[objectives_projection.field_index],
            ],
            dim=1,
        )
        x_critic_add = self.critic_linear_add(x_critic_input)
        x_critic_max = self.critic_linear_max(x_critic_input)

        # snag global indicators
        critic_x_mp = torch.relu(
            global_max_pool(x_critic_max, objectives_projection.batch)
        )
        if self.is_recurrent:
            critic_x_mp, rnn_hxs = self._forward_gru(critic_x_mp, rnn_hxs, masks)
        # snag a summuation of complexity
        # graph norm = safe node sum
        critic_x_ap = self.critic_ap_norm(
            global_add_pool(x_critic_add, objectives_projection.batch)
        )
        critic_x_ap = torch.relu(critic_x_ap)
        critic_x = torch.cat([critic_x_mp, critic_x_ap], dim=1)
        critic_value = self.critic_gate(critic_x)

        self.last_tensors["x"] = x
        self.last_tensors["trunk"] = trunk
        self.last_tensors["action_votes"] = action_votes
        self.last_tensors["critic_x"] = critic_x

        batch_votes = []
        batch_size = dom.batch.max().item() + 1
        for i in range(batch_size):
            _m = action_batch_idx == i
            # print(_m.shape)
            shares = action_votes[_m]
            batch_votes.append(shares)
        # print("bv", batch_votes[0].shape)
        return (critic_value, batch_votes, rnn_hxs)


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
