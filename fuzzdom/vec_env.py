from torch_geometric.transforms import Distance
from torch_geometric.data import Data, Batch
import gym
import numpy as np
import networkx as nx
from collections import defaultdict
import asyncio

from .domx import short_embed
from .state import MiniWoBGraphState
from .data import vectorize_projections, from_networkx


def minmax_scale(x, min, max):
    if x is None:
        return 0.0
    return (x - min) / (max - min)


def tuplize(f):
    def thunk(v):
        return (f(v),)

    return thunk


def one_hot(i, k):
    oh = np.zeros((k,), np.float32)
    oh[i] = 1
    return oh


def state_to_vector(graph_state: MiniWoBGraphState, prior_actions: dict):
    e_dom = encode_dom_graph(graph_state.dom_graph)
    e_fields = encode_fields(graph_state.fields)
    fields_projection_data = vectorize_projections(
        {"field": list(range(len(e_fields.nodes)))},
        e_dom,
        source_domain="dom",
        final_domain="dom_field",
    )
    e_leaves, leaves = encode_dom_leaves(e_dom)
    actions_data = vectorize_projections(
        {
            "ux_action": [
                {"action_idx": i, "action_one_hot": one_hot(i, 5)} for i in range(4)
            ],
            "field": [{"field_idx": i} for i in e_fields.nodes],
            "leaf": [{"dom_idx": e_dom.nodes[i]["dom_idx"]} for i in leaves],
        },
        e_dom,
        source_domain="dom",
        final_domain="action",
    )
    e_history = encode_prior_actions(e_dom, prior_actions, graph_state.fields)
    dom_data, fields_data = map(from_networkx, [e_dom, e_fields])
    leaves_data = from_networkx(
        e_leaves, dom_index=dom_data.num_nodes, leaf_index=len(e_leaves)
    )
    history_data = from_networkx(
        e_history,
        ux_action_index=4,
        dom_index=dom_data.num_nodes,
        field_index=fields_data.num_nodes,
    )
    # because history is usually empty
    history_data.num_nodes = len(e_history)
    assert leaves_data.num_nodes

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

        # TODO action_idx & action ?
        o.add_node(i, key=ke, query=ve, field_idx=(i,), index=i, order=(order,))
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


def encode_dom_leaves(dom_graph: nx.DiGraph):
    """
    For each leaf node we create a new graph with distances relative to the leaf
    and the values are replaced with a mask
    """
    pa = nx.DiGraph()
    leaves = list(filter(lambda n: dom_graph.out_degree(n) == 0, dom_graph.nodes))
    for i, k in enumerate(leaves):
        for u, dom_node in dom_graph.nodes(data=True):
            node = {
                "index": pa.number_of_nodes(),
                "leaf_index": i,
                "mask": (1,) if u == k else (0,),
                "dom_idx": dom_node["dom_idx"],
                "dom_index": dom_node["dom_idx"],
                "origin_length": (0.0,),
                "action_lenth": (0.0,),  # nx.shortest_path_length(dom_graph, k, u),
            }
            pa.add_node(f"{k}-{u}", **node)
            for (_u, v) in filter(lambda y: y[0] == u, dom_graph.edges(u)):
                pa.add_edge(f"{k}-{u}", f"{k}-{v}")
    return pa, leaves


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
        dom, objectives, obj_projection, leaves, actions, history = (
            self.last_observation
        )
        node_idx = action[0]
        action_id = actions.action_idx[node_idx].item()
        dom_idx = actions.dom_idx[node_idx].item()
        field_idx = actions.field_idx[node_idx].item()
        dom_ref = list(self.last_state.dom_graph.nodes)[dom_idx]
        field_value = list(self.last_state.fields.values)[field_idx]
        # n = {
        #    k: self.last_observation[k][node_idx]
        #    for k in self.last_observation.keys
        #    if not k.startswith("edge_")
        # }
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
