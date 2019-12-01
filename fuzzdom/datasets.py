import re
import os
import os.path as osp
import torch
from torch_geometric.data import Dataset
import json
import gzip
import networkx as nx
from torch_geometric.utils import from_networkx

from .domx import json_to_graph, miniwob_to_graph
from .vec_env import encode_dom_graph, encode_fields
from .env import WebInterface
from .state import MiniWoBGraphState, fields_factory


class DomDataset(Dataset):
    def __init__(self, root, urls, transform=None, pre_transform=None):
        super(DomDataset, self).__init__(root, transform, pre_transform)
        self.urls = urls

    @property
    def raw_file_names(self):
        return [n for n in os.listdir(self.raw_dir) if n.endswith(".json")]

    @property
    def processed_file_names(self):
        return [n for n in os.listdir(self.processed_dir) if n.endswith(".pt")]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        self.web_interface = WebInterface()
        driver = self.web_interface._driver
        for url in self.urls:
            print(url)
            i = hash(url)
            outpath = osp.join(self.raw_dir, f"{i}.json")
            if osp.exists(outpath):
                continue
            if url.startswith("file:/"):
                assert os.path.exists(url[7:]), url[7:]
            driver.get(url)
            self.web_interface.wait_for_dom()
            self.web_interface._injection_check()
            # print(self.web_interface.location)
            dom_struct = self.web_interface.visible_dom
            # print(driver.get_log("browser"))
            # print(driver.get_log("driver"))
            assert driver.page_source
            assert self.web_interface.html, str(driver.page_source)

            try:
                data = json.dumps(dom_struct)
            except Exception as e:
                print(e)
                continue
            open(outpath, "w").write(data)
        # Download to `self.raw_dir`.

    def process(self):
        i = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            dom_json = json.load(open(raw_path, "r"))
            dom_nx = json_to_graph(dom_json)
            if not len(dom_nx) or not len(dom_nx.edges):
                print("Empty", raw_path)
                continue
            data = encode_dom_graph(dom_nx)
            data = from_networkx(data)
            data.x = torch.cat(
                [
                    data[key]
                    for key in [
                        "text",
                        "value",
                        "tag",
                        "classes",
                        "rx",
                        "ry",
                        "width",
                        "height",
                        "top",
                        "left",
                    ]
                ],
                dim=1,
            )
            data.test_mask = torch.FloatTensor(data.x.shape[0]).uniform_() > 0.8
            data.train_mask = ~data.test_mask
            data.val_mask = torch.FloatTensor(data.x.shape[0]).uniform_() > 0.9

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, osp.join(self.processed_dir, "data_{}.pt".format(i)))
            i += 1

    def get(self, idx):
        if idx >= len(self):
            raise IndexError
        data = torch.load(osp.join(self.processed_dir, "data_{}.pt".format(idx)))
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
        return [n for n in os.listdir(self.processed_dir) if n.endswith(".pt")]

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
            if replay_json["rawReward"] <= 0 or not replay_json["utterance"]:
                continue
            try:
                fields = fields_factory(self.task_name, replay_json["utterance"])
            except ValueError as e:
                print(e)
                continue
            fields_e = encode_fields(fields)
            fields_matrix = from_networkx(fields_e)

            frames = []
            states = replay_json["states"]
            current_string = ""
            last_dom_nx = None
            for j, state in enumerate(states):
                dom_json = state["dom"]
                dom_nx = miniwob_to_graph(dom_json)
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
                data = encode_dom_graph(dom_nx, current_node)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                frame = {
                    "dom": from_networkx(data),
                    "fields": fields_matrix,
                    "action": action_str,
                    "field_id": field_id,
                    "node_id": list(last_dom_nx).index(current_node["ref"])
                    if current_node and last_dom_nx
                    else None,
                    "time": state["time"],
                    "reward": 0.0,
                }
                frames.append(frame)
                last_dom_nx = dom_nx
            if len(frames) > 1:
                frames[-1]["reward"] = replay_json["rawReward"]
                torch.save(
                    frames, osp.join(self.processed_dir, "data_{}.pt".format(p_idx))
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
    from glob import glob
    from .dir_paths import MINIWOB_HTML, DATA_DIR
    import linkGrabber
    import re

    download = True
    paths = []
    if download:
        paths = [
            f"file://{f}" for f in glob(MINIWOB_HTML + "/**/*.html", recursive=True)
        ]
        paths.reverse()
        initial_domains = [
            "www.ebay.com",
            "www.twitter.com",
            "www.netflix.com",
            "www.wikipedia.org",
            "www.youtube.com",
            "www.google.com",
            "www.airbnb.com",
            # "www.hotels.com",
            # "www.amazon.com",
            "www.cars.com",
            "www.twitch.tv",
            "store.steampowered.com",
            "www.reuters.com",
            # "imgur.com",
            # "www.lowes.com",
            "www.cbssports.com",
            "www.nfl.com",
            # "www.expedia.com",
            "www.walmart.com",
            "www.wayfair.com",
            "bing.com",
            # "reddit.com",
        ]
        for u in initial_domains:
            print(u)
            links = linkGrabber.Links(f"https://{u}/")
            l = links.find(limit=100, href=re.compile("//" + u), duplicates=False)
            _c = lambda x: x if x.startswith("http") else "https:" + x
            paths.extend([_c(e["href"]) for e in l])
    d = DomDataset(DATA_DIR+"/dom-dataset", paths)
    if download:
        d.download()
    d.process()
