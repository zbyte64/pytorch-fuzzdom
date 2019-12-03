from PIL import Image
from copy import copy
import networkx as nx

from miniwob.fields import Fields, get_field_extractor

from .domx import add_image_slices_to_graph, short_embed


def fields_factory(task, utterance):
    if isinstance(utterance, dict):
        fields = Fields(utterance["fields"])
    else:
        fields = get_field_extractor(task)(utterance)
        fields._d.pop("dummy", None)
        if not len(fields):
            # assert False, str(task)
            fields._d["objective"] = utterance
    return fields


class MiniWoBGraphState(object):
    """MiniWoB Graph state.
    """

    def __init__(
        self,
        utterance: str,
        fields,
        dom_graph: nx.DiGraph,
        screenshot: Image,
    ):
        assert isinstance(dom_graph, nx.DiGraph)
        self.utterance = utterance
        self.fields = fields
        self.dom_graph = dom_graph
        if screenshot:
            add_image_slices_to_graph(self.dom_graph, screenshot)
        self.screenshot = screenshot
        self.clipboard_text = ""
        """
        print(self.dom_nodes)
        print(self.dom_node_to_idx)
        for node in self.dom_nodes:
            print(node, list(self.dom_graph.successors(node)))
        """

    def copy_node_text(self, ref):
        self = copy(self)
        node_info = self.dom_graph.nodes[ref]
        self.clipboard_text = node_info.get("text", "")
        return self
