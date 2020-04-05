from torch_geometric.utils import from_networkx
from torch_geometric.transforms import Distance
import gym
import numpy as np
import networkx as nx
from collections import defaultdict

from .domx import short_embed
from .state import MiniWoBGraphState


def minmax_scale(x, min, max):
    if x is None:
        return 0.0
    return (x - min) / (max - min)


def tuplize(f):
    def thunk(v):
        return (f(v),)

    return thunk


def state_to_vector(graph_state: MiniWoBGraphState, prior_actions: dict):
    e_dom = encode_dom_graph(graph_state.dom_graph)
    e_fields = encode_fields(graph_state.fields)
    e_actions = encode_dom_actionables(e_dom)
    e_history = encode_prior_actions(e_dom, prior_actions, graph_state.fields)
    e_dom, e_fields, e_actions, e_history = map(
        nx.convert_node_labels_to_integers, [e_dom, e_fields, e_actions, e_history]
    )
    dom_data, fields_data, actions_data, history_data = map(
        from_networkx, [e_dom, e_fields, e_actions, e_history]
    )
    import torch

    dom_data.edge_index, fields_data.edge_index, actions_data.edge_index, history_data.edge_index = map(
        lambda x: x.type(torch.LongTensor),
        [
            dom_data.edge_index,
            fields_data.edge_index,
            actions_data.edge_index,
            history_data.edge_index,
        ],
    )

    return (dom_data, fields_data, actions_data, history_data)


def encode_fields(fields):
    o = nx.DiGraph()
    n = len(fields._d)
    for i, (key, value) in enumerate(fields._d.items()):
        ke = short_embed(key)
        ve = short_embed(value)
        order = (i + 1) / n
        # TODO action_idx & action ?
        o.add_node(i, key=ke, query=ve, field_idx=(i,), order=(order,))
        if i:
            o.add_edge(i - 1, i)
    return o


def encode_prior_actions(dom_graph: nx.DiGraph, prior_actions: dict, fields):
    history = nx.DiGraph()
    field_keys = list(fields.keys)
    for dom_ref, revisions in prior_actions.items():
        for revision, (action_idx, node, field_idx) in enumerate(revisions):
            action = ["click", "paste_field", "copy", "paste"][action_idx]
            field_key = field_keys[field_idx]
            k = f"{dom_ref}-{field_key}-{action}"
            if k in dom_graph:
                l = f"{dom_ref}-{field_key}-{action}-{revision}"
                history.add_node(
                    l,
                    dom_idx=dom_ref,
                    field_idx=field_idx,
                    action_idx=action_id,
                    revision=revision,
                    action=action,
                )
    return history


def encode_dom_actionables(dom_graph: nx.DiGraph):
    """
    For each leaf node we create a new graph with distances relative to the leaf
    and the values are replaced with a mask
    """
    potential_actions = nx.DiGraph()
    leaves = filter(lambda n: dom_graph.out_degree(n) == 0, dom_graph.nodes)
    for i, k in enumerate(leaves):
        for u in list(dom_graph.nodes):
            node = dom_graph.nodes[u]
            node = {
                "mask": 1 if u == k else 0,
                "dom_idx": node["dom_idx"],
                "origin_length": 0,
                "action_lenth": 0,  # nx.shortest_path_length(dom_graph, k, u),
            }
            l = f"{i}-{u}"
            potential_actions.add_node(l, **node)
            for (_u, v) in filter(lambda y: y[0] == u, dom_graph.edges(u)):
                m = f"{i}-{v}"
                potential_actions.add_edge(l, m)
    return potential_actions


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
    for i, node in enumerate(g.nodes):
        encoded_data = dict()
        for key, f in encode_with.items():
            encoded_data[key] = f(g.nodes[node].get(key))
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
        assert len(self.last_observation.dom_idx.shape) == 2, str(
            self.last_observation.dom_idx.shape
        )
        node_idx = action[0]
        action_id = self.last_observation.action_idx[node_idx].item()
        dom_idx = self.last_observation.dom_idx[node_idx].item()
        field_idx = self.last_observation.field_idx[node_idx].item()
        dom_ref = list(self.last_state.dom_graph.nodes)[dom_idx]
        field_value = list(self.last_state.fields.values)[field_idx]
        n = {
            k: self.last_observation[k][node_idx]
            for k in self.last_observation.keys
            if not k.startswith("edge_")
        }
        self.prior_actions[dom_ref].append((action_id, n, field_idx))
        return (action_id, dom_ref, field_value)

    def observation(self, obs: MiniWoBGraphState):
        assert isinstance(obs, MiniWoBGraphState)
        self.last_state = obs
        obs = state_to_vector(obs, self.prior_actions)
        self.last_observation = obs
        return obs

    def step(self, action):
        observation, reward, done, info = self.env.step(self.action(action))
        self.step_result((observation, reward, done, info))
        return self.observation(observation), reward, done, info

    def step_result(self, result):
        observation, reward, done, info = result
        if done or info.get("task_done"):
            self.prior_actions = defaultdict(list)

    def reset(self):
        self.prior_actions = defaultdict(list)
        obs = self.env.reset()
        return self.observation(obs)


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
