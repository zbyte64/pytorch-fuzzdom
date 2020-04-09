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
    SGConv,
    global_mean_pool,
)
import torch.nn.functional as F
from torch_geometric.utils import softmax

from a2c_ppo_acktr.model import NNBase
from a2c_ppo_acktr.utils import init


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
        init_t = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), 2 ** 0.5
        )
        init_r = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.dom_encoder = dom_encoder

        dom_attr_size = 5
        action_one_hot_size = 5
        query_input_dim = 6 + 2 * dom_attr_size
        if dom_encoder:
            query_input_dim += dom_encoder.out_channels
        self.attr_norm = nn.BatchNorm1d(text_embed_size)
        self.global_conv = GCNConv(query_input_dim, hidden_size - query_input_dim)

        # self.history_att_conv = graph_conv(5, 1)
        self.leaves_att_conv = SAGEConv(hidden_size + 3, 1)
        # objective pos + attr similarities
        dom_objective_size = hidden_size + 1 + 8
        self.objective_att_conv = SAGEConv(dom_objective_size, 1)

        self.dom_tag = nn.Sequential(
            init_t(nn.Linear(text_embed_size, dom_attr_size)), nn.Tanh()
        )
        self.dom_classes = nn.Sequential(
            init_t(nn.Linear(text_embed_size, dom_attr_size)), nn.Tanh()
        )
        self.query_key_ux_action_att = init_(
            nn.Linear(text_embed_size, action_one_hot_size)
        )
        self.leaf_ux_action_att = SAGEConv(hidden_size + 3, action_one_hot_size)
        self.order_bias_w = nn.Parameter(torch.tensor(1.0))

        # multiplied and maxpooled to become our actor gate(objective, leaves)
        self.actor_conv_objective = AGNNConv()
        self.actor_conv_leaf = AGNNConv()
        self.critic_ap_conv = SAGEConv(dom_objective_size, hidden_size)
        self.critic_mp_conv = SAGEConv(dom_objective_size, hidden_size)
        self.critic_ap_norm = nn.BatchNorm1d(hidden_size)

        self.critic_linear = init_(nn.Linear(hidden_size * 2, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        from torch_geometric.data import Batch, Data

        assert isinstance(inputs, tuple), str(type(inputs))
        assert all(map(lambda x: isinstance(x, Batch), inputs))
        dom, objectives, objectives_projection, leaves, actions, history = inputs
        # actions.original_batch
        assert actions.edge_index is not None, str(inputs)

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
                    dom.edge_index,
                )
            )
            x = torch.cat([x, _add_x], dim=1)

        _add_x = torch.relu(self.global_conv(x, dom.edge_index))
        x = torch.cat([x, _add_x], dim=1)

        if False and history.num_nodes:
            # process history into seperate attention scores and cast onto final action graph
            history_att = self.history_att_conv(
                history.action_one_hot, history.edge_index
            )
            history_att_dom = global_max_pool(history_att, history.dom_index)
            print(history.dom_index)
            print(history_att_dom.shape)
            print(actions.dom_leaf_index.shape)
            history_att_dom = history_att_dom[actions.dom_leaf_index]
            history_att_ux = global_max_pool(history_att, history.ux_action_index)[
                actions.ux_action_index
            ]
            history_att_obj = global_max_pool(history_att, history.field_index)[
                actions.field_index
            ]
        else:
            history_att_dom = 1
            history_att_ux = 1
            history_att_obj = 1

        # leaves are targetable elements
        x_leaves = torch.cat(
            [
                x[leaves.dom_index],
                leaves.origin_length,
                leaves.action_length,
                leaves.mask.type(torch.float),
            ],
            dim=1,
        )
        # compute a leaf dependent dom attention mask to indicate relevent nodes
        leaves_att_raw = self.leaves_att_conv(x_leaves, leaves.edge_index)
        leaves_att = softmax(leaves_att_raw, leaves.leaf_index)
        # infer action from leaf
        leaf_ux_action_att = F.softmax(
            global_max_pool(
                self.leaf_ux_action_att(x_leaves * leaves_att, leaves.edge_index),
                leaves.leaf_index,
            ),
            dim=1,
        ).view(-1, 1)[leaves.leaf_index]

        # compute an objective dependent dom attention mask
        # start with word embedding similarities
        obj_sim = lambda tag, obj: F.cosine_similarity(
            dom[tag][objectives_projection.dom_index],
            objectives[obj][objectives_projection.field_index],
        ).view(-1, 1)
        x_obj_input = torch.cat(
            [
                x[objectives_projection.dom_index],
                obj_sim("text", "key"),
                obj_sim("text", "query"),
                obj_sim("value", "key"),
                obj_sim("value", "query"),
                obj_sim("tag", "key"),
                obj_sim("tag", "query"),
                obj_sim("classes", "key"),
                obj_sim("classes", "query"),
                objectives.order[objectives_projection.field_index],
            ],
            dim=1,
        )
        x_obj = self.objective_att_conv(x_obj_input, objectives_projection.edge_index)
        x_obj_att = torch.relu(
            x_obj
        )  # softmax(x_obj, objectives_projection.field_index)
        # infer action from query key
        obj_ux_action_att = F.softmax(
            self.query_key_ux_action_att(self.attr_norm(objectives.key)), dim=1
        ).view(-1, 1)

        self.last_x_att = torch.cat(
            [
                leaves_att[actions.dom_leaf_index],
                x_obj_att[objectives_projection.dom_index[actions.dom_field_index]],
            ],
            dim=1,
        )
        # "fill in" with attentional propagation
        _aco = self.actor_conv_objective(
            x_obj_att[objectives_projection.dom_index[actions.dom_field_index]],
            actions.edge_index,
        )
        _acf = self.actor_conv_leaf(
            leaves_att[actions.dom_leaf_index], actions.edge_index
        )
        # x_actions intersects objectives and leaf attention
        x_actions = _aco * _acf
        # give bias by order
        x_order = 1 - self.order_bias_w * objectives.order[actions.field_index]
        action_potentials = (
            x_actions
            * obj_ux_action_att[actions.ux_action_index]
            * leaf_ux_action_att[actions.ux_action_index]
            + x_order
        )
        action_votes = torch.relu(
            global_add_pool(action_potentials, actions.action_index)
        )

        # critic is aware of objectives & dom
        x_critic_input = x_obj_input.detach()
        # snag global indicators
        critic_x_mp = self.critic_mp_conv(
            x_critic_input, objectives_projection.edge_index
        )
        critic_x_mp = torch.relu(
            global_max_pool(critic_x_mp, objectives_projection.batch)
        )
        if self.is_recurrent:
            critic_x_mp, rnn_hxs = self._forward_gru(critic_x_mp, rnn_hxs, masks)
        # snag a summuation of complexity
        critic_x_ap = self.critic_ap_conv(
            x_critic_input, objectives_projection.edge_index
        )
        critic_x_ap = global_add_pool(critic_x_ap, objectives_projection.batch)
        critic_x_ap = torch.tanh(self.critic_ap_norm(critic_x_ap))
        critic_x = torch.cat([critic_x_mp, critic_x_ap], dim=1)

        self.last_x = x
        self.last_x_actions = x_actions
        self.last_action_votes = action_votes
        self.last_critic_x = critic_x

        batch_votes = []
        batch_size = dom.batch.max().item() + 1
        for i in range(batch_size):
            _m = actions.batch == i
            # print(_m.shape)
            batch_votes.append(action_votes[actions.action_index[_m]])
        # print("bv", batch_votes[0].shape)
        return (self.critic_linear(critic_x), batch_votes, rnn_hxs)


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
