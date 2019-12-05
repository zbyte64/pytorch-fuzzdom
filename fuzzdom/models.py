import torch
from torch import nn
from torch_geometric.nn import (
    SAGEConv,
    global_max_pool,
    GlobalAttention,
    GCNConv
)
import torch.nn.functional as F

from a2c_ppo_acktr.model import NNBase
from a2c_ppo_acktr.utils import init


class GNNBase(NNBase):
    def __init__(self, input_dim, dom_encoder=None, recurrent=False, hidden_size=64):
        super(GNNBase, self).__init__(recurrent, hidden_size, hidden_size)

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

        # attr cosine distances + visual bytes + graph position + attr embeds + action embedding
        query_input_dim = 4 * 2 + 8 + 2 + 5 * 3 + 2
        if dom_encoder:
            query_input_dim += dom_encoder.out_channels
        conv_size = hidden_size

        self.attr_norm = nn.BatchNorm1d(25)

        self.global_conv = SAGEConv(query_input_dim, conv_size - query_input_dim)

        # TODO num-actions + 1
        self.action_embedding = nn.Sequential(nn.Embedding(5, 2), nn.Tanh())
        self.field_key = nn.Sequential(init_t(nn.Linear(25, 5)), nn.Tanh())
        self.dom_tag = nn.Sequential(init_t(nn.Linear(25, 5)), nn.Tanh())
        self.dom_classes = nn.Sequential(init_t(nn.Linear(25, 5)), nn.Tanh())
        self.inputs_att_gate = nn.Sequential(init_r(nn.Linear(conv_size, 1)), nn.ReLU())
        self.inputs_attention = GlobalAttention(self.inputs_att_gate)

        """self.main = nn.Sequential(
            init_r(nn.Linear(conv_size * 2, hidden_size)), nn.ReLU()
            #nn.Linear(conv_size * 2, hidden_size, bias=False), nn.Tanh()
        )
        """
        self.main = nn.Bilinear(conv_size, conv_size, hidden_size)

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        from torch_geometric.data import Batch, Data

        assert isinstance(inputs, Batch), str(type(inputs))

        query_dom_attr = lambda attr: torch.cat(
            [
                F.cosine_similarity(inputs.key, inputs[attr]).unsqueeze(1),
                F.cosine_similarity(inputs.query, inputs[attr]).unsqueeze(1),
                # F.pairwise_distance(inputs.key, inputs[attr]).unsqueeze(1),
                # F.pairwise_distance(inputs.query, inputs[attr]).unsqueeze(1),
            ],
            dim=1,
        )

        query = torch.cat(
            [
                query_dom_attr("text"),
                query_dom_attr("value"),
                query_dom_attr("tag"),
                query_dom_attr("classes"),
            ],
            dim=1,
        )

        x = torch.cat(
            [
                query,
                self.field_key(self.attr_norm(inputs.key)),
                self.dom_tag(self.attr_norm(inputs.tag)),
                self.dom_classes(self.attr_norm(inputs.classes)),
                inputs.rx,
                inputs.ry,
                inputs.width,
                inputs.height,
                inputs.top,
                inputs.left,
                inputs.focused,
                inputs.depth,
                inputs.order,
                inputs.tampered,
                self.action_embedding(inputs.action_idx + 1).squeeze(1),
            ],
            dim=1,
        ).clamp(-1, 1)

        if self.dom_encoder:
            # CONSIDER: this should be done before taking the cartesian product
            _add_x = torch.tanh(
                self.dom_encoder(
                    torch.cat(
                        [
                            inputs.text,
                            inputs.value,
                            inputs.tag,
                            inputs.classes,
                            inputs.rx,
                            inputs.ry,
                            inputs.width,
                            inputs.height,
                            inputs.top,
                            inputs.left,
                        ],
                        dim=1,
                    ),
                    inputs.edge_index,
                )
            )
            x = torch.cat([x, _add_x], dim=1)

        x_size = x.shape[1]

        _add_x = torch.tanh(self.global_conv(x, inputs.edge_index))
        _x = torch.cat([x, _add_x], dim=1)

        if self.is_recurrent:
            masks = masks[inputs.batch]
            if rnn_hxs.shape[0] == 1:
                rnn_hxs = torch.cat([rnn_hxs] * inputs.batch.shape[0])
            else:
                rnn_hxs = rnn_hxs[inputs.batch]
            _x, rnn_hxs = self._forward_gru(_x, rnn_hxs, masks)
            rnn_hxs = global_max_pool(rnn_hxs, inputs.batch)

        self.last_x = x
        self.last_query = query

        # zero out non-leaf nodes
        not_leaf_mask = (inputs.order < 0).squeeze()
        _x[not_leaf_mask] = 0

        # critic actor split here
        global_pool = global_max_pool(_x, inputs.batch)

        inputs_at = self.inputs_attention(_x, inputs.batch)
        self.last_inputs_at = inputs_at

        x = torch.tanh(self.main(inputs_at, global_pool))

        # emit node_id, and field_id attention
        inputs_votes = self.inputs_att_gate(_x) + 1e-15
        inputs_votes[not_leaf_mask] = 0.0

        batch_votes = []
        batch_size = inputs.batch.max().item() + 1
        for i in range(batch_size):
            _m = inputs.batch == i
            batch_votes.append(inputs_votes[_m])

        return (self.critic_linear(x), batch_votes, rnn_hxs)


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
