from torch_geometric.data import Batch, Data


class TupleBatch:
    @staticmethod
    def from_data_list(data_list):
        known_types = {}
        # data_list is a list of tuples

        for i, entry in enumerate(data_list):
            for j, data in enumerate(entry):
                for key in data.keys:
                    _type = data[key].type()
                    if key in known_types:
                        assert known_types[key] == _type, key
                    else:
                        known_types[key] = _type
        return tuple(map(Batch.from_data_list, zip(*data_list)))
