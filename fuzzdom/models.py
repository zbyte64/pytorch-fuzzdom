import torch
from torch import nn
from torch_geometric.nn import (
    SAGEConv,
    GraphConv,
    global_max_pool,
    global_add_pool,
    GlobalAttention,
    GCNConv,
    AGNNConv,
    GATConv,
    ARMAConv,
    TAGConv,
    SGConv,
    global_mean_pool,
    BatchNorm,
    GraphSizeNorm,
    InstanceNorm,
    APPNP,
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

reverse_edges = lambda e: torch.stack([e[1], e[0]]).contiguous()
full_edges = lambda e: torch.cat(
    [add_self_loops(e)[0], reverse_edges(e)], dim=1
).contiguous()
# full_edges = lambda x: add_self_loops(x)[0]
# reverse_edges = lambda x: x


class AdditiveMask(nn.Module):
    def __init__(self, input_dim, mask_dim=1, K=5):
        super(AdditiveMask, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.x_fn = nn.Sequential(init_t(nn.Linear(input_dim, input_dim)), nn.Tanh(),)
        # self.conv = SAGEConv(mask_dim, mask_dim)
        # self.conv = GCNConv(mask_dim, mask_dim)
        self.conv = APPNP(K=K, alpha=self.alpha)
        # self.conv = GraphConv(mask_dim, mask_dim)
        self.K = K

    def forward(self, x, mask, edge_index):
        x = self.x_fn(x)
        # edge_weights = self.pdist(x[edge_index[0]], x[edge_index[1]]).view(
        #    edge_index.shape[1]
        # )
        # edge_weights = self.edge_weight_fn(dist * directions).view(-1)
        edge_weights = torch.relu(
            F.cosine_similarity(x[edge_index[0]], x[edge_index[1]])
        ).view(-1)
        fill = torch.relu(mask)
        # for i in range(self.K):
        #    add_fill = fill[edge_index[0]] * edge_weights

        fill = self.conv(fill, edge_index, edge_weights)
        # fill = torch.tanh(fill)
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
            init_r(nn.Linear(text_embed_size, dom_attr_size)), nn.ReLU(),
        )
        self.dom_classes = nn.Sequential(
            init_r(nn.Linear(text_embed_size, dom_attr_size)), nn.ReLU(),
        )
        self.dom_fn = nn.Sequential(
            init_t(nn.Linear(query_input_dim, hidden_size - query_input_dim)),
            nn.Tanh(),
        )
        self.global_conv = SAGEConv(hidden_size, hidden_size)
        # self.global_conv = SGConv(hidden_size, hidden_size, K=7)
        # self.x_dom_norm = BatchNorm(hidden_size)

        self.dom_ux_action_fn = nn.Sequential(
            init_t(nn.Linear(hidden_size, action_one_hot_size)), nn.Softmax(dim=1),
        )
        self.leaves_conv = AdditiveMask(hidden_size, 1, K=7)
        # attr similarities
        attr_similarity_size = 8
        self.objective_conv = AdditiveMask(hidden_size, 1, K=7)
        self.objective_int_fn = nn.Sequential(
            init_t(nn.Linear(attr_similarity_size, 1)), nn.Sigmoid(),
        )

        self.objective_ux_action_fn = nn.Sequential(
            init_r(nn.Linear(text_embed_size * 2, text_embed_size)),
            nn.ReLU(),
            init_t(nn.Linear(text_embed_size, action_one_hot_size)),
            nn.Softmax(dim=1),
        )
        # fn and maxpooled to become our actor gate(objective, leaves)
        # leaf mask, obj mask, order, [dom,obj]ux_actions
        # trunk_size = 1 + attr_similarity_size + 1 + 2
        # ux similarity, dom intereset, order
        trunk_size = 3
        self.pdist = nn.PairwiseDistance(keepdim=True)
        self.actor_gate = nn.Sequential(nn.Linear(trunk_size, 1, bias=False), nn.ReLU())

        critic_size = hidden_size + action_one_hot_size + attr_similarity_size
        self.critic_obj_fn = nn.Sequential(
            init_r(nn.Linear(text_embed_size * 2, text_embed_size)),
            nn.ReLU(),
            init_r(nn.Linear(text_embed_size, action_one_hot_size)),
            nn.ReLU(),
        )
        self.critic_linear_add = nn.Sequential(
            init_r(nn.Linear(critic_size, hidden_size)), nn.ReLU()
        )
        self.critic_linear_max = nn.Sequential(
            init_r(nn.Linear(critic_size, hidden_size)), nn.ReLU()
        )
        self.critic_ap_norm = nn.BatchNorm1d(hidden_size)
        self.critic_gate = nn.Sequential(init_(nn.Linear(hidden_size * 2, 1)),)
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
                self.dom_tag(self.attr_norm(dom.tag)),
                self.dom_classes(self.attr_norm(dom.classes)),
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

        # x = torch.relu(self.x_dom_norm(torch.cat([x, _add_x], dim=1)))
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
        self.last_tensors["leaves_edge_weights"] = leaves_ew

        # infer action from dom node
        dom_ux_action = self.dom_ux_action_fn(x)

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
        obj_att_input = self.objective_int_fn(obj_tag_similarities)
        self.last_tensors["obj_att_input"] = obj_att_input
        self.last_tensors["obj_tag_similarities"] = obj_tag_similarities
        obj_att, obj_ew = self.objective_conv(
            proj_x[objectives_projection.dom_index],
            obj_att_input,
            full_edges(objectives_projection.edge_index),
        )

        self.last_tensors["obj_edge_weights"] = obj_ew
        # infer action from query key
        obj_ux_action = self.objective_ux_action_fn(
            torch.cat(
                [self.attr_norm(objectives.key), self.attr_norm(objectives.query)],
                dim=1,
            )
        )

        # project into main trunk
        ux_action_consensus = (
            obj_ux_action.view(-1, 1)[actions.ux_action_index]
            * dom_ux_action.view(-1, 1)[actions.dom_ux_action_index]
        ).view(-1, 1)
        dom_interest = (
            obj_att[objectives_projection.field_index][actions.dom_field_index]
            * leaves_att[actions.dom_leaf_index]
        )
        self.last_tensors["ux_action_consensus"] = ux_action_consensus
        self.last_tensors["dom_interest"] = dom_interest
        trunk = torch.cat(
            [dom_interest, ux_action_consensus, objectives.order[actions.field_index],],
            dim=1,
        )
        action_votes = global_max_pool(self.actor_gate(trunk), actions.action_index)

        self.last_tensors["obj_att"] = obj_att
        self.last_tensors["leaves_att"] = leaves_att
        self.last_tensors["obj_ux_action"] = obj_ux_action
        self.last_tensors["dom_ux_action"] = dom_ux_action

        # critic is aware of objectives & dom
        critic_obj = self.critic_obj_fn(
            torch.cat(
                [self.attr_norm(objectives.key), self.attr_norm(objectives.query)],
                dim=1,
            )
        )
        x_critic_input = torch.cat(
            [
                proj_x[objectives_projection.dom_index],
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
            _m = actions.batch == i
            # print(_m.shape)
            batch_votes.append(action_votes[actions.action_index][_m])
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
