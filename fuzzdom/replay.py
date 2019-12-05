import gym
import glob
import os
import random
import torch
import gzip
import json
from os import path as osp
import networkx as nx
from torch_geometric.data import Dataset
from collections import defaultdict

from .dir_paths import DATA_DIR
from .state import MiniWoBGraphState, fields_factory
from .vec_env import state_to_vector
from .domx import miniwob_to_graph


class ReplayRepository:
    def __init__(self, glob_str, root=DATA_DIR+"/replays"):
        self.glob_str = glob_str
        self.folders = glob.glob(glob_str)
        self.task_datasets = {}
        task_roots = {}
        task_folders = defaultdict(list)
        for folder_name in self.folders:
            task_name = os.path.split(folder_name)[-1]
            d_root = os.path.join(root, task_name)
            if not os.path.exists(d_root):
                os.makedirs(d_root)
            task_folders[task_name].append(folder_name)
            task_roots[task_name] = d_root
        for task_name, sources in task_folders.items():
            dataset = MiniWobReplayDataset(
                task_name=task_name, source_dirs=sources, root=task_roots[task_name]
            )
            self.task_datasets[task_name] = dataset

    def get_task_replay(self, task_name):
        dataset = self.task_datasets.get(task_name)
        if not dataset:
            return
        idx = random.randint(0, len(dataset) - 1)
        return dataset[idx]

    def get_dataset(self):
        filenames = []
        for task_name, ds in self.task_datasets.items():
            root = ds.root
            filenames.extend(ds.processed_file_names)
        return RawDataset(root, filenames)


class RawDataset(Dataset):
    def __init__(self, root, filenames, transform=None, pre_transform=None):
        self.filenames = filenames
        super(RawDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.filenames

    def __len__(self):
        return len(self.filenames)

    def download(self):
        pass

    def process(self):
        pass

    def get(self, idx):
        data = torch.load(self.filenames[idx])
        return data


class MiniWobReplayDataset(Dataset):
    def __init__(
        self, root, task_name, source_dirs=None, transform=None, pre_transform=None
    ):
        self.task_name = task_name
        self.source_dirs = source_dirs or self.raw_dir
        if isinstance(self.source_dirs, str):
            self.source_dirs = [self.source_dirs]
        super(MiniWobReplayDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [
            os.path.join(d, n)
            for d in self.source_dirs
            for n in os.listdir(d)
            if n.endswith(".gz") or n.endswith(".json")
        ]

    @property
    def processed_file_names(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        return [osp.join(self.processed_dir, n) for n in os.listdir(self.processed_dir) if n.endswith(".pt")]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        pass

    def process(self):
        p_idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            if raw_path.endswith(".json"):
                with open(raw_path, "r") as fin:
                    replay_json = json.load(fin)
            else:
                with gzip.GzipFile(raw_path, "r") as fin:
                    replay_json = json.load(fin)
            utterance = replay_json["utterance"]
            states = replay_json["states"]
            if replay_json["rawReward"] <= 0 or not utterance or len(states) < 2:
                continue
            try:
                fields = fields_factory(self.task_name, utterance)
            except ValueError as e:
                print(e)
                continue

            frames = []
            current_string = ""
            last_dom_nx = last_data = None
            for j, state in enumerate(states):
                dom_json = state["dom"]
                dom_nx = miniwob_to_graph(dom_json)
                graph_state = MiniWoBGraphState(utterance, fields, dom_nx, None)
                assert len(dom_nx)
                action_str = "wait"
                action = state["action"]
                current_node = None
                field_id = None
                # skip unrecognized actions
                if action and action["type"] not in ("click", "keypress"):
                    continue
                if (
                    action and action["timing"] != 1
                ):  # timing indicates that the same action has multiple phases
                    if action["type"] == "click":
                        action_str = "click"
                    if action["type"] == "keypress":
                        code = action["keyCode"] or action["charCode"]
                        char = chr(code)
                        current_string += char
                        # look at next event
                        if (
                            len(states) > j + 2
                            and states[j + 1]["action"] == "keypress"
                        ):
                            # goto next action
                            continue
                        action_str = "paste_field"
                        current_string = apply_backspaces(current_string)
                        # identify field
                        for i, (key, value) in enumerate(fields._d.items()):
                            if value == current_string:
                                field_id = i
                                break
                        current_string = ""
                    elif current_string:
                        # TODO identify field
                        print("Warning: uncaught transition from keypress")
                        current_string = ""
                    # TODO offset by scroll
                    if "x" in action and last_dom_nx:
                        # find current node
                        x, y = action["x"], action["y"]
                        # returns a path for depth first search, reverse order to get leaves first
                        t = list(nx.dfs_tree(dom_nx))
                        t.reverse()
                        for node_id in t:
                            node = dom_nx.nodes[node_id]
                            # print("Node", node_id, node)
                            if not node or node_id not in last_dom_nx:
                                continue
                            if (
                                x > node["top"]
                                and y > node["left"]
                                and x < node["top"] + node["height"]
                                and y < node["left"] + node["width"]
                            ):
                                current_node = node
                                break

                if current_node is not None and field_id is None:
                    t = current_node.get("text")
                    for i, (key, value) in enumerate(fields._d.items()):
                        if value == t:
                            field_id = i
                            break

                data = state_to_vector(graph_state)
                node_idx = -1
                if current_node and last_dom_nx:
                    last_nodes = list(last_dom_nx.nodes)
                    #field_values = list(fields.values)
                    for idx in range(last_data.dom_idx.shape[0]):
                        _dom_idx = last_data.dom_idx[idx]
                        _field_idx = last_data.field_idx[idx]
                        _action_str = ["click", 'paste_field', "copy", "paste", "wait"][last_data.action_idx[idx]]
                        if _action_str != action_str:
                            continue
                        _dom_ref = last_nodes[_dom_idx]
                        #_field_value = field_values[_field_idx]
                        if _dom_ref == current_node["ref"]:
                            if field_id is None or field_id == _field_idx:
                                node_idx = idx
                                break

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                reward = 0.0
                if j + 1 == len(states):
                    reward = replay_json["rawReward"]
                frame = (data, node_idx, reward)
                last_dom_nx = dom_nx
                last_data = data
                torch.save(
                    frame, osp.join(self.processed_dir, "data_{}.pt".format(p_idx))
                )
                p_idx += 1

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, "data_{}.pt".format(idx)))
        return data


def apply_backspaces(s):
    while "\x08" in s:
        s = re.replace("[^\x08]\x08", "", s)
    return s


if __name__ == "__main__":
    r = ReplayRepository("/code/miniwob-plusplus-demos/*turk/*")
    for t, d in r.task_datasets.items():
        print(t)
        d.download()
        d.process()
