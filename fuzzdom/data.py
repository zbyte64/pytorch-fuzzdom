import torch
from torch_geometric.data import Batch, Data


class SubData(Data):
    def __inc__(self, key, value):
        if key == "subbatch":
            return self.__num_graphs__
        if key == "node_idx":
            return self.__num_dom_nodes__
        return Data.__inc__(self, key, value)


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
