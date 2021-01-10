import torch
from torch_geometric.transforms import (
    Compose,
    FaceToEdge,
    Distance,
    Spherical,
    KNNGraph,
    AddSelfLoops,
)
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, add_self_loops, contains_self_loops
import gym
import numpy as np
import networkx as nx
from collections import defaultdict
import asyncio
import scipy.spatial
import random
from itertools import chain

from .domx import short_embed, text_embed_size
from .state import MiniWoBGraphState, DomInfo
from .data import vectorize_projections, from_networkx, SubData, broadcast_edges


def minmax_scale(x, min, max):
    if x is None:
        return 0.0
    return (x - min) / (max - min)


def tuplize(f):
    def thunk(v):
        return (f(v),)

    return thunk


def one_hot(i, k):
    oh = torch.zeros((k,), dtype=torch.float32)
    oh[i] = 1
    return oh


def replace_nan(x, v=0):
    x[torch.isnan(x)] = v
    return x


class FixEdgeAttr:
    def __call__(self, data):
        data.edge_attr = replace_nan(data.edge_attr)
        return data


class Delaunay:
    make_fully_connected = KNNGraph(k=3)

    def __call__(self, data):
        pos = data.pos.cpu().numpy()
        assert len(pos.shape) == 2 and pos.shape[1] == 3, str(pos.shape)
        if pos.shape[0] > 4:
            tri = scipy.spatial.Delaunay(pos, qhull_options="Qt Qbb Qc Qz Qx Q12")
            face = torch.from_numpy(tri.simplices)
            assert len(face.shape) == 2
            assert face.shape[1] == 4
            edge_index = torch.cat(
                [
                    face[:, 0:2],
                    face[:, 1:3],
                    face[:, 2:4],
                    torch.cat([face[:, 3:4], face[:, 0:1]], dim=1),
                ],
                dim=0,
            ).t()
            edge_index = edge_index.to(data.pos.device, torch.long)
            assert edge_index.shape[0] == 2, str(edge_index.shape)
            data.edge_index = edge_index
        else:
            # for small graphs
            data = self.make_fully_connected(data)
        return data


class ToUndirected:
    def __call__(self, data):
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        return data


class DomDirection:
    def __call__(self, data):
        direction = data.depth[data.edge_index[0]] > data.depth[data.edge_index[1]]
        data.edge_attr = direction.type(torch.float32).view(-1, 1)
        return data


SPATIAL_TRANSFORMER = Compose(
    [Delaunay(), ToUndirected(), AddSelfLoops(), Distance(), Spherical(), FixEdgeAttr()]
)

DOM_TRANSFORMER = Compose([ToUndirected(), DomDirection(), FixEdgeAttr()])


def chomp_leaves(leaves, k=20, **kwargs):
    if len(leaves) > k:
        print("Big dom", len(leaves))
        return random.sample(leaves, k)
    return leaves


def state_to_vector(graph_state: MiniWoBGraphState, filter_leaves=chomp_leaves):
    dom_data, leaves, max_depth = encode_dom_info(graph_state.dom_info)
    assert dom_data.num_nodes, str(graph_state.dom_info)
    e_fields = encode_fields(graph_state.fields)

    fields_data = from_networkx(e_fields)
    # prefix dom data edges and attributes with `dom` and `spatial`
    dom_data = DOM_TRANSFORMER(dom_data)
    dom_data.dom_edge_index = dom_data.edge_index
    dom_data.dom_edge_attr = dom_data.edge_attr
    dom_data.edge_attr = None
    dom_data = SPATIAL_TRANSFORMER(dom_data)
    dom_data.spatial_edge_index = dom_data.edge_index
    dom_data.spatial_edge_attr = dom_data.edge_attr

    fields_projection_data = vectorize_projections(
        {"field": list({"field_idx": i} for i in range(len(e_fields.nodes)))},
        dom_data,
        source_domain="dom",
        final_domain="dom_field",
    )
    fields_projection_data.spatial_edge_attr = fields_projection_data.edge_attr
    fields_projection_data.dom_edge_attr = dom_data.dom_edge_attr.repeat(
        len(e_fields.nodes), 1
    )
    fields_projection_data.spatial_edge_index = fields_projection_data.edge_index
    fields_projection_data.dom_edge_index = dom_data.dom_edge_index.repeat(
        1, len(e_fields.nodes)
    )
    broadcast_edges(
        fields_projection_data,
        dom_data.spatial_edge_index,
        len(e_fields.nodes),
        "br_spatial_edge_index",
    )
    broadcast_edges(
        fields_projection_data,
        dom_data.dom_edge_index,
        len(e_fields.nodes),
        "br_dom_edge_index",
    )
    leaves = filter_leaves(
        leaves, dom=dom_data, field=fields_data, dom_field=fields_projection_data
    )
    assert leaves
    leaves_data = vectorize_projections(
        {"leaf": list({"dom_idx": i} for i in leaves)},
        dom_data,
        source_domain="dom",
        final_domain="dom_leaf",
    )
    # mask is leaf nodes
    leaves_data.mask = (leaves_data.dom_idx == leaves_data.dom_index).view(-1, 1)
    # prefix leaves data edges and attributes with `dom` and `spatial`
    leaves_data.spatial_edge_attr = leaves_data.edge_attr
    leaves_data.dom_edge_attr = dom_data.dom_edge_attr.repeat(len(leaves), 1)
    broadcast_edges(
        leaves_data, dom_data.spatial_edge_index, len(leaves), "br_spatial_edge_index"
    )
    broadcast_edges(
        leaves_data, dom_data.dom_edge_index, len(leaves), "br_dom_edge_index"
    )
    leaves_data.spatial_edge_index = leaves_data.edge_index
    leaves_data.dom_edge_index = dom_data.dom_edge_index.repeat(1, len(leaves))
    actions_data = vectorize_projections(
        {
            "ux_action": [
                {"action_idx": i, "action_one_hot": one_hot(i, 4)} for i in range(4)
            ],
            "field": [{"field_idx": i} for i in e_fields.nodes.keys()],
            "leaf": [{"dom_idx": torch.tensor(i, dtype=torch.int64)} for i in leaves],
        },
        dom_data,
        source_domain="dom",
        final_domain="action",
        add_intersections=[
            ("field", "ux_action"),
            ("leaf", "ux_action"),
            ("field", "dom"),
            ("leaf", "dom"),
        ],
    )

    logs_data = vectorize_logs(graph_state.logs)
    return (
        dom_data,
        fields_data,
        fields_projection_data,
        leaves_data,
        actions_data,
        logs_data,
    )


def vectorize_logs(logs: dict):
    # dictionary of arrays
    x = []
    for channel, entries in logs.items():
        k = torch.from_numpy(short_embed(channel))
        for entry in entries:
            l = torch.from_numpy(short_embed(str(entry)))
            x.append(torch.cat([k, l]))
        if not len(entries):
            l = torch.from_numpy(short_embed(""))
            x.append(torch.cat([k, l]))
    x = torch.stack(x)
    return Data(x=x, edge_index=torch.zeros(2, 0))


def encode_fields(fields):
    o = nx.DiGraph()
    n = len(fields._d)
    for i, (key, value) in enumerate(fields._d.items()):
        ke = short_embed(key)
        ve = short_embed(value)
        order = (i + 1) / n
        is_last = 1.0 if i + 1 == n else 0.0
        # TODO action_idx & action ?
        o.add_node(
            i,
            key=ke,
            query=ve,
            field_idx=(i,),
            index=i,
            order=(order,),
            is_last=(is_last,),
        )
        if i:
            o.add_edge(i - 1, i)
    return o


RADIO_VALUES = {True: 1.0, False: -1.0}


def iter_assign(t, iterable):
    for i, v in enumerate(iterable):
        t[i] = v
    return t


iter_tensor = lambda iterable, entries, size: iter_assign(
    torch.Tensor(entries, size), iterable
)


short_embed_t = lambda x: torch.from_numpy(short_embed(x))
safe_max = lambda x, m=[1]: max(chain(x, m))
truthy_embed = lambda x: 1.0 if x else 0.0
radio_embed = lambda x: RADIO_VALUES.get(x, 0.0)


def encode_dom_info(g: DomInfo, encode_with=None, text_embed_size=text_embed_size):
    assert len(g.row)
    o = {}
    leaves = []
    num_nodes = len(g.ref)
    assert num_nodes, str(g)
    # max_width = safe_max(g.width)
    # max_height = safe_max(g.height)
    max_y = safe_max(g.top)
    max_x = safe_max(g.left)
    # max_ry = safe_max(map(abs, g.ry))
    # max_rx = safe_max(map(abs, g.rx))
    max_depth = safe_max(g.depth)
    if encode_with is None:
        encode_with = {
            "text": short_embed_t,
            "value": short_embed_t,
            "tag": short_embed_t,
            "classes": short_embed_t,
            # "rx": lambda x: minmax_scale(x, -max_rx, max_rx),
            # "ry": lambda x: minmax_scale(x, -max_ry, max_ry),
            # "height": lambda x: minmax_scale(x, 0, max_height),
            # "width": lambda x: minmax_scale(x, 0, max_width),
            # "top": lambda x: minmax_scale(x, 0, max_y),
            # "left": lambda x: minmax_scale(x, 0, max_x),
            "focused": truthy_embed,
            "tampered": truthy_embed,
            "depth": lambda x: minmax_scale(x, 0, max_depth),
        }
    text_keys = {"text", "value", "tag", "classes"}
    for key, f in encode_with.items():
        size = text_embed_size if key in text_keys else 1
        o[key] = iter_tensor(map(f, getattr(g, key)), num_nodes, size)
    o["radio_value"] = iter_tensor(map(radio_embed, g.value), num_nodes, 1)
    o["index"] = torch.arange(0, num_nodes)
    o["dom_index"] = o["index"]
    o["dom_ref"] = torch.tensor(g.ref).view(-1, 1)
    o["pos"] = torch.Tensor(num_nodes, 3)
    for i, (top, left, width, height, depth, n_children) in enumerate(
        zip(g.top, g.left, g.width, g.height, g.depth, g.n_children)
    ):
        iter_assign(
            o["pos"][i],
            (
                (left + width / 2) / max_x,  #
                (top + height / 2) / max_y,  #
                depth / max_depth,  #
            ),
        )
        if n_children == 0:
            leaves.append(i)

    o["edge_index"] = torch.tensor([g.row, g.col], dtype=torch.long)
    o["edge_index"] = torch.cat(
        (o["edge_index"], torch.arange(0, num_nodes).repeat(2, 1)), dim=1
    )
    data = SubData(o, index=num_nodes)
    data.num_nodes = num_nodes
    return data, leaves, max_depth


class ReceiptsGymWrapper(gym.ObservationWrapper):
    """
    Store complex observations in a receipt factory
    """

    observation_space = gym.spaces.Discrete(1)  # np.inf

    def __init__(self, env, receipt_factory):
        gym.ObservationWrapper.__init__(self, env)
        self.receipt_factory = receipt_factory

    def observation(self, obs):
        idx = self.receipt_factory(obs)
        return idx


class GraphGymWrapper(gym.Wrapper):
    """
    Convert graph state to tensor/Data
    """

    action_space = gym.spaces.Discrete(1)  # np.inf
    observation_space = gym.spaces.Discrete(np.inf)

    def __init__(self, env, filter_leaves=chomp_leaves):
        super().__init__(env)
        self.filter_leaves = filter_leaves

    def action(self, action):
        assert len(action) == 1, str(action)
        (dom, objectives, obj_projection, leaves, actions, *_) = self.last_observation
        combination_idx = action[0]
        # ux_action, field, leaf
        selected_combo = actions.combinations[combination_idx]
        # print("Selected combo:", selected_combo)
        selected_targets = {k: v for i, c in selected_combo for k, v in c.items()}
        # print("Selected:", selected_targets)
        action_id = selected_targets["action_idx"]
        field_idx = selected_targets["field_idx"]
        dom_idx = selected_targets["dom_idx"]
        dom_ref = self.last_state.dom_info.ref[dom_idx]
        field_value = list(self.last_state.fields.values)[field_idx]
        assert isinstance(field_value, str), str(self.last_state.fields)
        return (action_id, dom_ref, field_value)

    def observation(self, obs: MiniWoBGraphState):
        assert isinstance(obs, MiniWoBGraphState), str(type(obs))
        self.last_state = obs
        obs = state_to_vector(obs, self.filter_leaves)
        self.last_observation = obs
        return obs

    async def exec_observation(self, obs, executor):
        assert isinstance(obs, MiniWoBGraphState), str(type(obs))
        loop = asyncio.get_event_loop()
        v_obs = await loop.run_in_executor(
            executor, state_to_vector, obs, self.filter_leaves
        )
        self.last_observation = v_obs
        self.last_state = obs
        return v_obs

    def step(self, action):
        observation, reward, done, info = self.env.step(self.action(action))
        return self.observation(observation), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    async def exec_reset(self, executor):
        obs = self.env.reset()
        return await self.exec_observation(obs, executor)


def make_vec_envs(envs, receipts, inner=lambda x: x, filter_leaves=chomp_leaves):
    from .asyncio_vector_env import AsyncioVectorEnv

    envs = [
        ReceiptsGymWrapper(
            inner(GraphGymWrapper(env, filter_leaves=filter_leaves)),
            receipt_factory=receipts,
        )
        for env in envs
    ]
    vec_env = AsyncioVectorEnv(envs)
    vec_env.observations = np.zeros((len(envs), 1), dtype=np.int32) - 1
    vec_env.action_space = gym.spaces.Discrete(1)
    return vec_env
