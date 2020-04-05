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

        # attr cosine distances + visual bytes + graph position + attr embeds + action embedding + revision
        query_input_dim = 4 * 2 + 6 + 2 + 5 * 3 + 2 + 1
        if dom_encoder:
            query_input_dim += dom_encoder.out_channels
        self.attr_norm = nn.BatchNorm1d(text_embed_size)

        assert hidden_size > query_input_dim + 16
        self.global_conv = SAGEConv(query_input_dim, hidden_size - query_input_dim)

        # TODO num-actions + 1
        self.action_embedding = nn.Sequential(nn.Embedding(5, 2), nn.Tanh())
        self.field_key = nn.Sequential(init_t(nn.Linear(text_embed_size, 5)), nn.Tanh())
        self.dom_tag = nn.Sequential(init_t(nn.Linear(text_embed_size, 5)), nn.Tanh())
        self.dom_classes = nn.Sequential(
            init_t(nn.Linear(text_embed_size, 5)), nn.Tanh()
        )
        self.actor_gate = nn.Sequential(
            init_r(nn.Linear(hidden_size * 3, hidden_size)),
            nn.ReLU(),
            init_r(nn.Linear(hidden_size, 1)),
            nn.ReLU(),
        )
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
        dom, objectives, actions, history = inputs

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
                inputs.width * inputs.height,
                inputs.top * inputs.left,
                inputs.focused,
                inputs.depth,
                inputs.order,
                inputs.tampered,
                inputs.revision,
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

        # drop non-leaf nodes
        leaf_mask = (inputs.actionable == 1).squeeze()
        # assert leaf_mask.sum().item()
        _x = _x[leaf_mask]
        _batch = inputs.batch[leaf_mask]

        # critic input is pooled indicators
        global_mp = global_max_pool(_x, _batch)
        if self.is_recurrent:
            global_mp, rnn_hxs = self._forward_gru(global_mp, rnn_hxs, masks)
        node_mp = global_mp[_batch]
        critic_input = torch.cat([_x, node_mp], dim=1).detach()
        critic_add = self.critic_add_gate(critic_input)
        global_ap = torch.tanh(global_add_pool(torch.relu(critic_add), _batch))
        critic_x = torch.cat([global_mp.detach(), global_ap], dim=1)

        # actor input is node actions with indicator inputs
        actor_x = torch.cat([_x, node_mp, critic_add.detach()], dim=1)

        self.last_x = x
        self.last_query = query

        self.last_inputs_at = actor_x

        # emit node_id, and field_id attention
        inputs_votes = self.actor_gate(actor_x)

        batch_votes = []
        batch_size = _batch.max().item() + 1
        all_votes = torch.zeros(x.shape[0])
        all_votes.masked_scatter_(leaf_mask, inputs_votes)
        for i in range(batch_size):
            _m = inputs.batch == i
            batch_votes.append(all_votes[_m])

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
