import torch
import networkx as nx
import itertools
from torch_geometric.utils import from_networkx as _from_networkx
from torch_geometric.data import Batch, Data


def from_networkx(g, **indexes):
    data = SubData(_from_networkx(g), **indexes)
    data.num_nodes = g.number_of_nodes()
    return data


class SubData(Data):
    """
    Data meant to be packed with other data of different sizes
    pass in index sizes to auto increment while batching
    """

    def __init__(self, data_set, **foreign_keys):
        keys = data_set.keys
        if callable(keys):
            keys = keys()
        kwargs = {
            key: data_set[key].view(-1)
            if key.endswith("index") and not key.startswith("edge_")
            else data_set[key]
            for key in keys
        }
        Data.__init__(self, **kwargs)
        for key, num in foreign_keys.items():
            setattr(self, f"__{key}__", num)

    def __inc__(self, key, value):
        if hasattr(self, f"__{key}__"):
            return getattr(self, f"__{key}__")
        return Data.__inc__(self, key, value)


def vectorize_projections(
    projections: dict, source: nx.DiGraph, source_domain: str, final_domain: str
):
    """
    Assumes source already has integer labels!

    Encodes a new instance of `source` for each entry in `projections`

    projections: {proj_domain0: [items], proj_domain1: [items]...}
    """

    proj_domains = list(projections.keys())
    combinations = list(itertools.product(*map(enumerate, projections.values())))
    t_edge_index = torch.tensor(list(source.edges)).t().contiguous()
    num_nodes = source.number_of_nodes()

    data = {}
    edges = []
    index = -1

    for k, p in enumerate(combinations):
        # value w/ enum: [ (0, x0), (1, x1) ]
        # p: [(proj_index_0, proj_value_0), ...]
        k_edge_index = t_edge_index + k * num_nodes
        edges.append(k_edge_index)
        for u, src_node in source.nodes(data=True):
            index += 1
            node_index = src_node["index"]
            node = {
                "index": index,
                f"{final_domain}_index": k,
                f"{source_domain}_index": node_index,
            }
            for src_index, (proj_index, proj_value) in enumerate(p):
                p_domain = proj_domains[src_index]
                node[f"{source_domain}_{p_domain}_index"] = (proj_index + 1) * (
                    node_index + 1
                ) - 1
                node[f"{p_domain}_index"] = proj_index
                if isinstance(proj_value, dict):
                    node.update(proj_value)
                elif proj_value is not None:
                    node[f"{p_domain}_value"] = proj_value
            for key, value in node.items():
                data[key] = [value] if index == 0 else data[key] + [value]

    # return dst, combinations
    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data["edge_index"] = torch.cat(edges).view(2, -1)

    indexes = {
        f"{p_domain}_index": len(entries) for p_domain, entries in projections.items()
    }
    indexes[f"{final_domain}_index"] = len(combinations)
    indexes[f"{source_domain}_index"] = source.number_of_nodes()
    for p_domain, p_values in projections.items():
        indexes[f"{source_domain}_{p_domain}_index"] = num_nodes * len(p_values)

    data = SubData(data, **indexes)
    data.num_nodes = num_nodes * len(combinations)
    return data


def encode_projections(
    projections: dict, source: nx.DiGraph, source_domain: str, final_domain: str
):
    """
    Encodes a new instance of `source` for each entry in `projections`

    projections: {proj_domain0: [items], proj_domain1: [items]...}
    """
    index = -1
    dst = nx.DiGraph()
    proj_domains = list(projections.keys())
    combinations = list(itertools.product(*map(enumerate, projections.values())))
    for k, p in enumerate(combinations):
        # value w/ enum: [ (0, x0), (1, x1) ]
        # p: [(proj_index_0, proj_value_0), ...]
        for u, src_node in source.nodes(data=True):
            index += 1
            node_index = src_node["index"]
            node = {
                "index": index,
                f"{final_domain}_index": k,
                f"{source_domain}_index": node_index,
            }
            for src_index, (proj_index, proj_value) in enumerate(p):
                p_domain = proj_domains[src_index]
                node[f"{source_domain}_{p_domain}_index"] = (proj_index + 1) * (
                    node_index + 1
                ) - 1
                node[f"{p_domain}_index"] = proj_index
                if isinstance(proj_value, dict):
                    node.update(proj_value)
                elif proj_value is not None:
                    node[f"{p_domain}_value"] = proj_value
            dst.add_node(f"{k}-{u}", **node)
            for (_u, v) in filter(lambda y: y[0] == u, source.edges(u)):
                dst.add_edge(f"{k}-{u}", f"{k}-{v}")
    return dst, combinations


class TupleBatch:
    @staticmethod
    def from_data_list(data_list):
        # a batch of multiple samples
        ret = []
        for batch_idx, samples in enumerate(data_list):
            packed_sample = []
            ret.append(packed_sample)
            for source in samples:
                assert isinstance(source, Data)
                # why do we get floats?
                source.edge_index = source.edge_index.type(torch.long)
                packed_sample.append(source)
        return tuple(map(Batch.from_data_list, zip(*ret)))
