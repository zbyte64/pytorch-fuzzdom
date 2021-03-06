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
    encode_field_actions_onto_encoded_dom_graph(
        e_dom, e_fields, graph_state.fields, graph_state.utterance
    )
    encode_prior_actions_onto_encoded_dom_graph(
        e_dom, prior_actions, graph_state.fields
    )
    # for node_id in e_dom.nodes:
    #    print(node_id, e_dom.nodes[node_id])
    e_dom = nx.convert_node_labels_to_integers(e_dom)
    d = from_networkx(e_dom)
    return d


def encode_fields(fields):
    o = nx.DiGraph()
    n = len(fields._d)
    for i, (key, value) in enumerate(fields._d.items()):
        ke = short_embed(key)
        ve = short_embed(value)
        order = (i + 1) / n
        o.add_node(i, key=ke, query=ve, field_idx=(i,), order=(order,))
        if i:
            o.add_edge(i - 1, i)
    return o


def encode_prior_actions_onto_encoded_dom_graph(
    dom_graph: nx.DiGraph, prior_actions: dict, fields
):
    field_keys = list(fields.keys)
    for dom_ref, revisions in prior_actions.items():
        for revision, (action_id, node, field_idx) in enumerate(revisions):
            action = ["click", "paste_field", "copy", "paste"][action_id]
            field_key = field_keys[field_idx]
            k = f"{dom_ref}-{field_key}-{action}"
            if k in dom_graph:
                dom_graph.nodes[k]["tampered"] = (1.0,)
                node["revision"] = ((revision + 1) / len(revisions),)
                node["actionable"] = (0,)
                l = f"{dom_ref}-{field_key}-{action}-{revision}"
                dom_graph.add_node(l, **node)
                dom_graph.add_edge(k, l)


def encode_field_actions_onto_encoded_dom_graph(
    dom_graph: nx.DiGraph, fields_graph: nx.DiGraph, fields, utterance: str
):
    default_leaf_info = {
        "action": short_embed(""),
        "key": short_embed("utterance"),
        "query": short_embed(utterance),
        "field_idx": (-1,),
        "order": (-1.0,),
        "action_idx": (-1,),
        "revision": (-1.0,),
        "actionable": (0,),
    }
    for x in list(dom_graph.nodes):
        node = dom_graph.nodes[x]
        node.update(default_leaf_info)
        is_leaf = dom_graph.out_degree(x) == 0
        if is_leaf:
            parents = list(dom_graph.predecessors(x))
            parent = parents[-1] if parents else None
            for y, field_key in enumerate(fields.keys):
                field = fields_graph.nodes[y]
                field_key = field_key.split()[0]
                action = {
                    "click": "click",
                    "select": "click",
                    "submit": "click",
                    "target": "click",
                    "copy": "copy",
                    "paste": "paste",
                }.get(field_key, "paste_field")
                action_idx = {"click": 0, "paste_field": 1, "copy": 2, "paste": 3}[
                    action
                ]
                k = f"{x}-{field_key}-{action}"
                field_node = dict(node)
                field_node.update(field)
                field_node["action"] = short_embed(action)
                field_node["action_idx"] = (action_idx,)
                field_node["actionable"] = (1,)
                dom_graph.add_node(k, **field_node)
                if parent is not None:
                    dom_graph.add_edge(parent, k)
                if action != "click":
                    action = "click"
                    k = f"{x}-{field_key}-{action}"
                    field_node = dict(node)
                    field_node.update(field)
                    field_node["action"] = short_embed(action)
                    field_node["action_idx"] = (0,)
                    field_node["actionable"] = (1,)
                    dom_graph.add_node(k, **field_node)
                    if parent is not None:
                        dom_graph.add_edge(parent, k)
            dom_graph.remove_node(x)


def encode_dom_graph(g: nx.DiGraph, encode_with=None):
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
            "tampered": tuplize(lambda x: 1.0 if x else 0.0),
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
