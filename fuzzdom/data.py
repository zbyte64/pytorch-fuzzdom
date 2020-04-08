import torch
import networkx as nx
import itertools
from collections import defaultdict
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

    data = defaultdict(list)
    edges = []
    index = -1
    # precompute key strings
    final_domain_key = f"{final_domain}_index"
    source_domain_key = f"{source_domain}_index"
    sp_keys = [
        {
            "sp_domain_index": f"{source_domain}_{p_domain}_index",
            "p_domain_index": f"{p_domain}_index",
        }
        for p_domain in proj_domains
    ]

    for k, p in enumerate(combinations):
        # value w/ enum: [ (0, x0), (1, x1) ]
        # p: [(proj_index_0, proj_value_0), ...]
        k_edge_index = t_edge_index + k * num_nodes
        edges.append(k_edge_index)
        for u, src_node in source.nodes(data=True):
            index += 1
            node_index = src_node["index"]
            data["index"].append(index)
            data[final_domain_key].append(k)
            data[source_domain_key].append(node_index)

            for keys, (proj_index, proj_value) in zip(sp_keys, p):
                data[keys["sp_domain_index"]].append(
                    (proj_index + 1) * (node_index + 1) - 1
                )
                data[keys["p_domain_index"]].append(proj_index)
                for key, value in proj_value.items():
                    data[key].append(value)

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data["edge_index"] = torch.cat(edges).view(2, -1)

    indexes = {
        keys["p_domain_index"]: len(entries)
        for keys, (p_domain, entries) in zip(sp_keys, projections.items())
    }
    indexes[final_domain_key] = len(combinations)
    indexes[source_domain_key] = source.number_of_nodes()
    for keys, (p_domain, p_values) in zip(sp_keys, projections.items()):
        indexes[keys["sp_domain_index"]] = num_nodes * len(p_values)

    data = SubData(data, **indexes)
    data.num_nodes = num_nodes * len(combinations)
    return data


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
