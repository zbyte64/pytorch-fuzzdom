import torch
from torch import nn
from torch_geometric.nn import (
    SAGEConv,
    global_max_pool,
    global_add_pool,
    GlobalAttention,
    GCNConv,
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
        super(GNNBase, self).__init__(recurrent, hidden_size, hidden_size)
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

        self.dom_encoder = dom_encoder

        query_input_dim = 6 + 2 * 5
        if dom_encoder:
            query_input_dim += dom_encoder.out_channels
        self.attr_norm = nn.BatchNorm1d(text_embed_size)

        assert hidden_size > query_input_dim + 16
        self.global_conv = SAGEConv(query_input_dim, hidden_size - query_input_dim)

        self.history_att_conv = SAGEConv(5, 1)
        self.dom_att_conv = SAGEConv(hidden_size + 3, 1)
        self.objective_att_query = init_t(nn.Linear(text_embed_size * 2, hidden_size))
        self.objective_att_conv = SAGEConv(hidden_size * 2 + 1, hidden_size)
        self.actions_conv = SAGEConv(hidden_size * 2 + 5, hidden_size)

        self.dom_tag = nn.Sequential(init_t(nn.Linear(text_embed_size, 5)), nn.Tanh())
        self.dom_classes = nn.Sequential(
            init_t(nn.Linear(text_embed_size, 5)), nn.Tanh()
        )
        self.actor_gate = nn.Sequential(init_r(nn.Linear(hidden_size, 1)), nn.ReLU())
        self.critic_add_gate = nn.Sequential(
            init_t(nn.Linear(hidden_size * 2, hidden_size)), nn.Tanh()
        )

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )
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

        global_add_x = self.global_conv(x, dom.edge_index)
        x = torch.cat([x, global_add_x], dim=1)

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

        # conv over dom+actions to produce action attention mask
        x_actions = x[actions.dom_index]
        x_leaves = torch.cat(
            [
                x[leaves.dom_index],
                leaves.origin_length,
                leaves.action_lenth,
                leaves.mask.type(torch.float),
            ],
            dim=1,
        )

        leaves_att_raw = self.dom_att_conv(x_leaves, leaves.edge_index)

        leaves_att = softmax(leaves_att_raw, leaves.leaf_index)
        x_actions = x_actions * leaves_att[actions.dom_leaf_index] * history_att_dom

        objective_att = self.objective_att_query(
            torch.cat([objectives.key, objectives.query], dim=1)
        )
        x_obj = torch.cat(
            [
                x[objectives_projection.dom_index],
                objectives.order[objectives_projection.field_index],
                objective_att[objectives_projection.field_index],
            ],
            dim=1,
        )
        x_obj = self.objective_att_conv(x_obj, objectives_projection.edge_index)

        x_actions = (
            x_actions
            * x_obj[objectives_projection.dom_index[actions.dom_field_index]]
            * history_att_obj
        )
        x_actions = torch.cat(
            [x_actions, actions.action_one_hot, objective_att[actions.field_index]],
            dim=1,
        )
        x_actions = self.actions_conv(x_actions, actions.edge_index)
        x_actions = x_actions * history_att_ux

        # print("x", x.shape)
        # print("x_actions", x_actions.shape)
        # print(actions.dom_index.shape)
        # print("objectives", objectives.batch.shape)

        action_potentials = global_max_pool(x_actions, actions.action_index)
        action_votes = self.actor_gate(action_potentials)
        # print("action_votes", action_votes.shape)

        # critic input is pooled indicators
        global_mp = global_max_pool(x, dom.batch)
        if self.is_recurrent:
            global_mp, rnn_hxs = self._forward_gru(global_mp, rnn_hxs, masks)
        node_mp = global_mp[dom.batch]
        critic_input = torch.cat([x, node_mp], dim=1).detach()
        critic_add = self.critic_add_gate(critic_input)
        global_ap = torch.tanh(global_add_pool(torch.relu(critic_add), dom.batch))
        critic_x = torch.cat([global_mp.detach(), global_ap], dim=1)

        self.last_x = x

        batch_votes = []
        batch_size = dom.batch.max().item() + 1
        # all_votes = torch.zeros(x.shape[0])
        # all_votes.masked_scatter_(leaf_mask, action_votes)
        """
        print("#" * 20)
        print(actions.action_index)
        print("actins_index", actions.action_index.shape)
        print("x_actions", x_actions.shape)
        print("action_potentials", action_potentials.shape)
        print("action_votes", action_votes.shape)
        print(actions.dom_field_index.shape)
        """
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
