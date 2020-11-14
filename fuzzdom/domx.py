import networkx as nx
from bs4 import BeautifulSoup as soup
from bpemb import BPEmb
import numpy as np
from PIL import Image
import random
from collections import defaultdict
from num2words import num2words
import re

text_embed_size = 300

bpemb_en = BPEmb(lang="en", dim=text_embed_size)
word_or_num_re = r"(?P<num>[0-9]+)|(?P<word>[\w]+)"


def _chunk_to_word(chunk):
    v = chunk.groupdict()
    s = v["word"]
    if v["num"] is not None:
        s = num2words(v["num"])
    return s


def embed(x, numeral_re=r"[0-9]+"):
    if x is not None:
        if not isinstance(x, str):
            x = str(x)
        # https://github.com/bheinzerling/bpemb/issues/20
        stream = map(_chunk_to_word, re.finditer(word_or_num_re, x))
        x = " ".join(stream)
        return bpemb_en.embed(x)


def short_embed(x):
    if x is not None:
        e = embed(x)
        if e.shape and e.shape[0]:
            return e.mean(0)
    return np.zeros(text_embed_size, dtype=np.float32)


def traverse_html(
    d: soup,
    g: nx.Graph,
    tag_map: dict = {
        "class": ["class"],
        "id": ["id"],
        "label": ["label", "name", "title", "placeholder"],
        "value": ["value"],
    },
    keep_empty: bool = False,
    parent=None,
) -> None:
    for count, i in enumerate(d.contents):
        if i.name is not None:
            i_str = len(g)
            attrs = dict()
            for dest, source in tag_map.items():
                entries = []
                for key in source:
                    if key in i.attrs:
                        value = i.attrs[key]
                        if isinstance(value, list):
                            entries.extend(value)
                        else:
                            entries.append(str(value))
                if len(entries):
                    value = " ".join(entries)
                    attrs[dest] = short_embed(value)
            if i.string:
                attrs["string"] = short_embed(i.string)
            if len(attrs) or len(i.contents) or keep_empty:
                for tag in tag_map.keys():
                    if tag not in attrs:
                        attrs[tag] = np.zeros(text_embed_size, dtype=np.float32)
                if "string" not in attrs:
                    attrs["string"] = np.zeros(text_embed_size, dtype=np.float32)
                attrs["tagname"] = short_embed(i.name)
                g.add_node(i_str, **attrs)
                if parent is not None:
                    g.add_edge(parent, i_str)
                traverse_html(i, g, tag_map, keep_empty, i_str)


def html_to_graph(html: str) -> nx.Graph:
    d = soup(html, "html.parser")
    full_graph = nx.Graph()
    traverse_html(d, full_graph)
    return full_graph


def traverse_json(
    d: dict,
    g: nx.DiGraph,
    tag_map: dict = {
        "classes": ["class", "type"],
        "id": ["id"],
        "label": ["label", "name", "title", "placeholder"],
        "value": ["value"],
    },
    keep_empty: bool = False,
    parent=None,
):
    if parent is None:
        parent = d["t"].lower()
    # t v a,c,d, f
    for count, i in enumerate(d["c"]):
        if i["a"].get("id"):
            i_str = "#" + i["a"].get("id")
        else:
            r = int(i["r"]) + 1
            i_str = f"{parent} > :nth-child({r})"
        attrs = dict()
        for dest, source in tag_map.items():
            entries = []
            for key in source:
                if key in i["a"]:
                    value = i["a"][key]
                    if isinstance(value, list):
                        entries.extend(value)
                    else:
                        entries.append(str(value))
            if len(entries):
                attrs[dest] = " ".join(entries)
        if i["v"]:
            attrs["text"] = i["v"]
        if len(attrs) or len(i["c"]) or keep_empty:
            for tag in tag_map.keys():
                if tag not in attrs:
                    attrs[tag] = None
            if "text" not in attrs:
                attrs["text"] = None
            attrs["tag"] = i["t"]
            if "type" in i["a"]:
                attrs["tag"] += "_" + i["a"]["type"]
            attrs["focused"] = i["f"]
            # add relative position values
            attrs["ry"] = i["d"]["top"] - d["d"]["top"]
            attrs["rx"] = i["d"]["left"] - d["d"]["left"]
            attrs["top"] = i["d"]["top"]
            attrs["left"] = i["d"]["left"]
            attrs["height"] = i["d"]["height"]
            attrs["width"] = i["d"]["width"]
            if i["a"].get("id"):
                attrs["ref"] = "document.getElementById('%s')" % i["a"].get("id")
            else:
                attrs["ref"] = "document.querySelector('%s')" % i_str
            # appease miniwob
            attrs["children"] = []
            attrs["tampered"] = False

            g.add_node(i_str, **attrs)
            if parent in g:
                g.add_edge(parent, i_str)
            traverse_json(i, g, tag_map, keep_empty, i_str)


INTERESTING_ATTRS = {
    "id",
    "label",
    "name",
    "title",
    "placeholder",
    "value",
    "type",
    "class",
}


def is_jdom_interesting(x):
    return x["v"] or (set(x["a"].keys()) & INTERESTING_ATTRS)


def json_to_graph(
    d: dict, to_keep=is_jdom_interesting, exclude=lambda x: False
) -> nx.DiGraph:
    def depth_filter(n):
        # prune any branches that dont pass `to_keep`
        if exclude(n):
            return False
        s = dict(n)
        # random.shuffle(n["c"])
        s["c"] = []
        added = False
        for child in n["c"]:
            r = depth_filter(child)
            if r:
                s["c"].append(r)
            added = added or bool(r)
        if not to_keep(n):
            if added:
                if len(s["c"]) == 1:
                    return s["c"][0]
                return s
            return False
        return s

    e = depth_filter(d)
    assert e, str(e)
    full_graph = nx.DiGraph()
    traverse_json(e, full_graph)
    return full_graph


def miniwob_to_graph(d: dict) -> nx.DiGraph:
    g = nx.DiGraph()

    def traverse(n, p=None):
        assert "ref" in n
        attrs = dict(n)
        # random.shuffle(attrs["children"])

        attrs["ry"] = n["top"]
        attrs["rx"] = n["left"]
        if p:
            attrs["ry"] -= p["top"]
            attrs["rx"] -= p["left"]
        g.add_node(n["ref"], **attrs)
        for child in attrs["children"]:
            traverse(child, n)
            g.add_edge(n["ref"], child["ref"])

    traverse(d)
    return g


def miniwob_to_dominfo(d: dict):
    from .state import DomInfo

    o = {key: [] for key in DomInfo._fields}

    def traverse(n, p=None, depth=0):
        assert "ref" in n
        i = len(o["ref"])
        attrs = dict(n)

        attrs["ry"] = n["top"]
        attrs["rx"] = n["left"]
        if p:
            attrs["ry"] -= p["top"]
            attrs["rx"] -= p["left"]
        attrs["n_children"] = len(attrs.get("children", []))
        attrs["depth"] = depth
        for key in [
            "n_children",
            "rx",
            "text",
            "top",
            "tag",
            "ry",
            "value",
            "height",
            "width",
            "depth",
            "classes",
            "focused",
            "tampered",
            "ref",
            "left",
        ]:
            value = attrs.get(key, None)
            o[key].append(value)
        o["row"].append(i)
        o["col"].append(i)
        for child in attrs.get("children", []):
            c = traverse(child, n, depth + 1)
            o["row"].append(i)
            o["col"].append(c)
        return i

    traverse(d)
    return o


def add_image_slices_to_graph(
    g: nx.DiGraph, img: Image, window_scroll=(0, 0), image_size=(48, 48)
):
    for node in g:
        d = node["dimensions"]
        box = (
            d["left"] + window_scroll[0],
            d["top"] + window_scroll[1],
            d["left"] + d["width"] + window_scroll[0],
            d["top"] + d["height"] + window_scroll[1],
        )
        subimg = img.crop(box).resize(image_size)
        node["img"] = np.asarray(subimg)


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    from torch_geometric.utils import from_networkx

    html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""
    full_graph = html_to_graph(html_doc)
    # nx.draw(full_graph, with_labels=True)
    # plt.show()
    print(list(full_graph.edges))
    d = from_networkx(full_graph)
    print(d)
    print(d.num_nodes)
    print(d["class"])
