from torch_geometric.utils import from_networkx
from torch_geometric.transforms import Distance
from torch_geometric.data import Data, Batch
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


class SubData(Data):
    def __init__(self, data_set, **foreign_keys):
        kwargs = {key: data_set[key] for key in data_set.keys}
        Data.__init__(self, **kwargs)
        for key, num in foreign_keys.items():
            setattr(self, f"__{key}__", num)

    def __inc__(self, key, value):
        if hasattr(self, f"__{key}__"):
            return getattr(self, f"__{key}__")
        return Data.__inc__(self, key, value)


def state_to_vector(graph_state: MiniWoBGraphState, prior_actions: dict):
    e_dom = encode_dom_graph(graph_state.dom_graph)
    e_fields = encode_fields(graph_state.fields)
    e_leaves = encode_dom_leaves(e_dom)
    e_actions = encode_dom_actionables(e_dom, graph_state.fields)
    e_history = encode_prior_actions(e_dom, prior_actions, graph_state.fields)
    dom_data, fields_data = map(from_networkx, [e_dom, e_fields])
    leaves_data = SubData(
        from_networkx(e_leaves), dom_index=dom_data.num_nodes, leaf_index=len(e_leaves)
    )
    actions_data = SubData(
        from_networkx(e_actions),
        dom_index=dom_data.num_nodes,
        field_index=fields_data.num_nodes,
        leaf_index=len(e_leaves),
        leaf_node_index=leaves_data.num_nodes,
        selection_index=len(e_leaves) * len(e_fields),
    )
    history_data = SubData(
        from_networkx(e_history),
        dom_index=dom_data.num_nodes,
        field_index=fields_data.num_nodes,
    )
    # because history is usually empty
    history_data.num_nodes = len(e_history)
    assert leaves_data.num_nodes
    assert len(e_actions)
    actions_data.num_nodes = len(e_actions)

    return (dom_data, fields_data, leaves_data, actions_data, history_data)


def encode_fields(fields):
    o = nx.DiGraph()
    n = len(fields._d)
    for i, (key, value) in enumerate(fields._d.items()):
        ke = short_embed(key)
        ve = short_embed(value)
        order = (i + 1) / n

        # TODO action_idx & action ?
        o.add_node(
            i, key=ke, query=ve, field_idx=(i,), field_index=(i,), order=(order,)
        )
        if i:
            o.add_edge(i - 1, i)
    return o


def encode_dom_actionables(dom_graph: nx.DiGraph, fields):
    """
    For each leaf & field emit a sub graph
    """
    pa = nx.DiGraph()
    leaves = filter(lambda n: dom_graph.out_degree(n) == 0, dom_graph.nodes)
    leaf_node_index = -1
    for i, k in enumerate(leaves):
        leaf = dom_graph.nodes[k]
        for u in list(dom_graph.nodes):
            leaf_node_index = +1
            for j, (key, value) in enumerate(fields._d.items()):
                node = dom_graph.nodes[u]
                pa.add_node(
                    f"{i}-{j}-{u}",
                    field_idx=j,
                    field_index=j,
                    action_idx=0,
                    dom_idx=leaf["dom_idx"],
                    dom_index=node["dom_idx"],
                    leaf_index=i,
                    leaf_node_index=leaf_node_index,
                    selection_index=(i + 1) * (j + 1) - 1,
                )
                for (_u, v) in filter(lambda y: y[0] == u, dom_graph.edges(u)):
                    pa.add_edge(f"{i}-{j}-{u}", f"{i}-{j}-{v}")

    return pa


def encode_prior_actions(dom_graph: nx.DiGraph, prior_actions: dict, fields):
    history = nx.DiGraph()
    field_keys = list(fields.keys)
    for dom_ref, revisions in prior_actions.items():
        for revision, (action_idx, node, field_idx) in enumerate(revisions):
            action = ["click", "paste_field", "copy", "paste"][action_idx]
            field_key = field_keys[field_idx]
            k = f"{dom_ref}-{field_key}-{action}-{revision}"
            history.add_node(
                k,
                dom_idx=dom_ref,
                dom_index=dom_ref,
                field_idx=field_idx,
                field_index=field_idx,
                action_idx=action_id,
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
    leaves = filter(lambda n: dom_graph.out_degree(n) == 0, dom_graph.nodes)
    for i, k in enumerate(leaves):
        for u in list(dom_graph.nodes):
            node = dom_graph.nodes[u]
            node = {
                "index": len(pa),
                "leaf_index": i,
                "mask": (1,) if u == k else (0,),
                "dom_idx": node["dom_idx"],
                "dom_index": node["dom_idx"],
                "origin_length": (0.0,),
                "action_lenth": (0.0,),  # nx.shortest_path_length(dom_graph, k, u),
            }
            pa.add_node(f"{k}-{u}", **node)
            for (_u, v) in filter(lambda y: y[0] == u, dom_graph.edges(u)):
                pa.add_edge(f"{k}-{u}", f"{k}-{v}")
    return pa


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
        encoded_data["dom_index"] = encoded_data["dom_idx"] = (i,)
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
        dom, objectives, leaves, actions, history = self.last_observation
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
        # self.prior_actions[dom_ref].append((action_id, n, field_idx))
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
