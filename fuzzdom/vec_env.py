import torch
from torch_geometric.transforms import Distance
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected, add_self_loops, contains_self_loops
import gym
import numpy as np
import networkx as nx
from collections import defaultdict
import asyncio

from .domx import short_embed
from .state import MiniWoBGraphState
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


def state_to_vector(graph_state: MiniWoBGraphState, prior_actions: dict):
    e_dom = encode_dom_graph(graph_state.dom_graph)
    e_fields = encode_fields(graph_state.fields)
    fields_projection_data = vectorize_projections(
        {"field": list({"field_idx": i} for i in range(len(e_fields.nodes)))},
        e_dom,
        source_domain="dom",
        final_domain="dom_field",
    )
    leaves_data, leaves = project_dom_leaves(e_dom)
    actions_data = vectorize_projections(
        {
            "ux_action": [{"action_idx": i} for i in range(4)],
            "field": [{"field_idx": i} for i in e_fields.nodes],
            "leaf": [
                {"dom_idx": torch.tensor(e_dom.nodes[i]["dom_idx"], dtype=torch.int64)}
                for i in leaves
            ],
        },
        e_dom,
        source_domain="dom",
        final_domain="action",
        add_intersections=[
            ("field", "ux_action"),
            ("leaf", "ux_action"),
            ("field", "dom"),
            ("leaf", "dom"),
        ],
    )

    e_history = encode_prior_actions(e_dom, prior_actions, graph_state.fields)
    dom_data, fields_data = map(from_networkx, [e_dom, e_fields])

    dom_data.dom_edge_index = dom_data.edge_index
    dom_data.spatial_edge_index = to_undirected(
        knn_graph(dom_data.pos, k=6, batch=None, loop=True, flow="source_to_target"),
        num_nodes=dom_data.num_nodes,
    )
    assert contains_self_loops(dom_data.spatial_edge_index)
    dom_data.edge_index = dom_data.spatial_edge_index
    dom_data = Distance(cat=False)(dom_data)

    history_data = from_networkx(
        e_history,
        ux_action_index=4,
        dom_index=dom_data.num_nodes,
        field_index=fields_data.num_nodes,
    )
    # because history is usually empty
    history_data.num_nodes = len(e_history)
    assert leaves_data.num_nodes

    fields_projection_data.dom_edge_index = fields_projection_data.edge_index
    fields_projection_data.edge_index = broadcast_edges(
        dom_data.spatial_edge_index,
        dom_data.num_nodes,
        len(fields_projection_data.combinations),
    )
    # assert contains_self_loops(dom_data.spatial_edge_index)
    # assert contains_self_loops(fields_projection_data.edge_index)
    fields_projection_data.edge_attr = dom_data.edge_attr.repeat(
        fields_projection_data.edge_index.shape[1]
        // dom_data.spatial_edge_index.shape[1],
        1,
    )

    return (
        dom_data,
        fields_data,
        fields_projection_data,
        leaves_data,
        actions_data,
        history_data,
    )


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


def encode_prior_actions(dom_graph: nx.DiGraph, prior_actions: [], fields):
    history = nx.DiGraph()
    field_keys = list(fields.keys)
    for revision, (action_idx, dom_idx, field_idx) in enumerate(prior_actions):
        action = ["click", "paste_field", "copy", "paste", "sleep"][action_idx]
        field_key = field_keys[field_idx]
        history.add_node(
            revision,
            dom_idx=dom_idx,
            dom_index=dom_idx,
            field_idx=field_idx,
            field_index=field_idx,
            action_idx=action_idx,
            action_one_hot=one_hot(action_idx, 5),
            ux_action_index=action_idx,
            revision=revision,
            action=action,
        )
    return history


def project_dom_leaves(source: nx.DiGraph):
    """
    For each leaf node we create a new graph with distances relative to the leaf
    and the values are replaced with a mask

    Also creates indexes:

    * dom_index (per source node)
    * leaf_index (per leaf)
    """
    # 1 out connection because self connected
    leaves = list(filter(lambda n: source.out_degree(n) == 0, source.nodes))

    t_edge_index = torch.tensor(list(source.edges)).t().contiguous()
    num_nodes = source.number_of_nodes()
    final_size = len(leaves) * num_nodes

    data = {
        "index": torch.zeros((final_size,), dtype=torch.int64),
        "leaf_index": torch.zeros((final_size,), dtype=torch.int64),
        "mask": torch.zeros((final_size, 1), dtype=torch.float32),
        "dom_idx": torch.zeros((final_size,), dtype=torch.int64),
        "dom_index": torch.zeros((final_size,), dtype=torch.int64),
        "dom_leaf_index": torch.zeros((len(leaves),), dtype=torch.int64),
    }

    edges = []
    index = -1

    for k, p in enumerate(leaves):
        k_edge_index = t_edge_index + k * num_nodes
        edges.append(k_edge_index)
        for u, src_node in source.nodes(data=True):
            index += 1
            node_index = src_node["index"]
            data["index"][index] = node_index
            data["leaf_index"][index] = k
            data["mask"][index] = 1 if u == p else 0
            data["dom_idx"][index] = src_node["dom_idx"][0]
            data["dom_index"][index] = node_index
        data["dom_leaf_index"][k] = torch.tensor(p, dtype=torch.int64)

    data["edge_index"] = torch.cat(edges).view(2, -1)
    data = SubData(
        data, dom_index=num_nodes, leaf_index=len(leaves), dom_leaf_index=num_nodes
    )
    data.num_nodes = final_size
    data.combinations = leaves
    return data, leaves


def encode_dom_graph(g: nx.DiGraph, encode_with=None):
    assert len(g)
    numeric_map = {}
    o = nx.DiGraph()
    if encode_with is None:
        max_width = max([g.nodes[n]["width"] for n in g] + [1.0])
        max_height = max([g.nodes[n]["height"] for n in g] + [1.0])
        max_y = max([g.nodes[n]["top"] for n in g] + [1.0])
        max_x = max([g.nodes[n]["left"] for n in g] + [1.0])
        max_ry = max([abs(g.nodes[n]["ry"]) for n in g] + [1.0])
        max_rx = max([abs(g.nodes[n]["rx"]) for n in g] + [1.0])
        encode_with = {
            "text": short_embed,
            "value": short_embed,
            "tag": short_embed,
            "classes": short_embed,
            "rx": tuplize(lambda x: minmax_scale(x, -max_rx, max_rx)),
            "ry": tuplize(lambda x: minmax_scale(x, -max_ry, max_ry)),
            "height": tuplize(lambda x: minmax_scale(x, 0, max_height)),
            "width": tuplize(lambda x: minmax_scale(x, 0, max_width)),
            "top": tuplize(lambda x: minmax_scale(x, 0, max_y)),
            "left": tuplize(lambda x: minmax_scale(x, 0, max_x)),
            "focused": tuplize(lambda x: 1.0 if x else 0.0),
        }
    d = nx.shortest_path_length(g, list(g.nodes)[0])
    max_depth = max(d.values()) + 1
    for i, (node, node_data) in enumerate(g.nodes(data=True)):
        encoded_data = dict()
        for key, f in encode_with.items():
            encoded_data[key] = f(node_data.get(key))
        encoded_data["index"] = i
        encoded_data["dom_idx"] = (i,)
        encoded_data["depth"] = ((d.get(node, 0) + 1) / max_depth,)
        encoded_data["pos"] = (
            encoded_data["rx"][0] + encoded_data["width"][0] / 2,
            encoded_data["ry"][0] + encoded_data["height"][0] / 2,
            encoded_data["depth"][0],
        )
        o.add_node(i, **encoded_data)
        numeric_map[node] = i
    for u, v in g.edges:
        o.add_edge(numeric_map[u], numeric_map[v])
    return o


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

    def action(self, action):
        assert len(action) == 1, str(action)
        (
            dom,
            objectives,
            obj_projection,
            leaves,
            actions,
            history,
        ) = self.last_observation
        combination_idx = action[0]
        # ux_action, field, leaf
        selected_combo = actions.combinations[combination_idx]
        # print("Selected combo:", selected_combo)
        selected_targets = {k: v for i, c in selected_combo for k, v in c.items()}
        # print("Selected:", selected_targets)
        action_id = selected_targets["action_idx"]
        field_idx = selected_targets["field_idx"]
        dom_idx = selected_targets["dom_idx"]
        dom_ref = list(self.last_state.dom_graph.nodes)[dom_idx]
        field_value = list(self.last_state.fields.values)[field_idx]
        self.prior_actions.append((action_id, dom_idx, field_idx))
        return (action_id, dom_ref, field_value)

    def observation(self, obs: MiniWoBGraphState):
        assert isinstance(obs, MiniWoBGraphState), str(type(obs))
        self.last_state = obs
        obs = state_to_vector(obs, self.prior_actions)
        self.last_observation = obs
        return obs

    async def exec_observation(self, obs, executor):
        assert isinstance(obs, MiniWoBGraphState), str(type(obs))
        loop = asyncio.get_event_loop()
        v_obs = await loop.run_in_executor(
            executor, state_to_vector, obs, self.prior_actions
        )
        self.last_observation = v_obs
        self.last_state = obs
        return v_obs

    def step(self, action):
        observation, reward, done, info = self.env.step(self.action(action))
        self.step_result((observation, reward, done, info))
        return self.observation(observation), reward, done, info

    def step_result(self, result):
        observation, reward, done, info = result
        if done or info.get("task_done"):
            self.prior_actions = []

    def reset(self):
        self.prior_actions = []
        obs = self.env.reset()
        return self.observation(obs)

    async def exec_reset(self, executor):
        self.prior_actions = []
        obs = self.env.reset()
        return await self.exec_observation(obs, executor)


def make_vec_envs(envs, receipts):
    from .asyncio_vector_env import AsyncioVectorEnv

    envs = [
        ReceiptsGymWrapper(GraphGymWrapper(env), receipt_factory=receipts)
        for env in envs
    ]
    vec_env = AsyncioVectorEnv(envs)
    vec_env.observations = np.zeros((len(envs), 1), dtype=np.int32) - 1
    vec_env.action_space = gym.spaces.Discrete(1)
    return vec_env
