import torch
import networkx as nx
import itertools
from torch_geometric.utils import from_networkx as _from_networkx
from torch_geometric.data import Batch, Data


def from_networkx(g, **indexes):
    data = SubData(_from_networkx(g), **indexes)
    data.num_nodes = len(g.nodes)
    return data


class SubData(Data):
    """
    Data meant to be packed with other data of different sizes
    pass in index sizes to auto increment while batching
    """

    def __init__(self, data_set, **foreign_keys):
        kwargs = {
            key: data_set[key].view(-1)
            if key.endswith("index") and not key.startswith("edge_")
            else data_set[key]
            for key in data_set.keys
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
    g, combinations = encode_projections(
        projections, source, source_domain, final_domain
    )
    indexes = {
        f"{p_domain}_index": len(entries) for p_domain, entries in projections.items()
    }
    indexes[f"{final_domain}_index"] = len(combinations)
    indexes[f"{source_domain}_index"] = len(source.nodes)
    for p_domain, p_values in projections.items():
        indexes[f"{source_domain}_{p_domain}_index"] = len(source.nodes) * len(p_values)

    data = from_networkx(g, **indexes)
    return data, g


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
        for u in list(source.nodes):
            index += 1
            src_node = source.nodes[u]
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
                if isinstance(source, Data):
                    # why do we get floats?
                    source.edge_index = source.edge_index.type(torch.long)
                    packed_sample.append(source)
                else:
                    # array of data
                    s = list(source)
                    if not s:
                        s = source
                    else:
                        s = Batch.from_data_list(s)
                    packed_sample.append(s)
        known_types = {}
        # data_list is a list of tuples

        for i, entry in enumerate(ret):
            for j, data in enumerate(entry):
                for key in data.keys:
                    _type = data[key].type()
                    if key in known_types:
                        assert known_types[key] == _type, key
                    else:
                        known_types[key] = _type
        return tuple(map(Batch.from_data_list, zip(*ret)))
