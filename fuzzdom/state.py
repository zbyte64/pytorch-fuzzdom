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
        dom_position: int,
        field_position: int,
    ):
        assert isinstance(dom_graph, nx.DiGraph)
        self.dom_position = dom_position
        self.field_position = field_position
        self.utterance = utterance
        self.fields = fields
        self.dom_graph = dom_graph
        self.dom_nodes = list(self.dom_graph)
        # print("G Size:", len(self.dom_nodes))
        # add_image_slices_to_graph(self.dom_graph, screenshot)
        self.dom_node_to_idx = {k: i for i, k in enumerate(self.dom_nodes)}
        if self.dom_position >= len(self.dom_graph):
            print("reseting dom position")
            self.dom_position = 0
        if self.field_position >= len(self.fields):
            self.field_position = 0
        self.screenshot = screenshot
        self.clipboard_text = ""
        """
        print(self.dom_nodes)
        print(self.dom_node_to_idx)
        for node in self.dom_nodes:
            print(node, list(self.dom_graph.successors(node)))
        """

    @property
    def current_node(self):
        if len(self.dom_graph.nodes) > self.dom_position:
            return self.dom_nodes[self.dom_position]

    @property
    def current_node_info(self):
        if len(self.dom_graph.nodes) > self.dom_position:
            return self.dom_graph.nodes[self.current_node]

    @property
    def current_dom_element(self):
        from miniwob.state import DOMElement

        return DOMElement(self.current_node_info)

    @property
    def current_field(self):
        if len(self.fields) > self.field_position:
            return list(self.fields.values)[self.field_position]

    @property
    def current_field_key(self):
        if len(self.fields) > self.field_position:
            return list(self.fields.keys)[self.field_position]

    @property
    def parent_node(self):
        p = list(self.dom_graph.predecessors(self.current_node))
        if p:
            return p[-1]
        return None

    @property
    def siblings_and_self(self):
        p = self.parent_node
        if p:
            return list(self.dom_graph.successors(p))

    def dom_up(self):
        p = list(self.dom_graph.predecessors(self.current_node))
        if not p:
            return self
        self = copy(self)
        self.dom_position = self.dom_node_to_idx[p[-1]]
        return self

    def dom_down(self):
        p = list(self.dom_graph.successors(self.current_node))
        if not p:
            return self
        self = copy(self)
        self.dom_position = self.dom_node_to_idx[p[0]]
        return self

    def dom_left(self):
        c = self.siblings_and_self
        if not c:
            return self
        if len(c) == 1:
            return self.dom_up()
        i = c.index(self.current_node)
        j = i - 1
        if j < 0:
            j = len(c) - 1
        self = copy(self)
        self.dom_position = self.dom_node_to_idx[c[j]]
        return self

    def dom_right(self):
        c = self.siblings_and_self
        if not c:
            return self
        if len(c) == 1:
            return self.dom_down()
        i = c.index(self.current_node)
        j = i + 1
        if j >= len(c):
            j = 0
        self = copy(self)
        self.dom_position = self.dom_node_to_idx[c[j]]
        return self

    def field_left(self):
        self = copy(self)
        self.field_position -= 1
        if self.field_position < 0:
            self.field_position = len(self.fields) - 1
        return self

    def field_right(self):
        self = copy(self)
        self.field_position += 1
        if self.field_position >= len(self.fields):
            self.field_position = 0
        return self

    def copy_node_text(self):
        self = copy(self)
        self.clipboard_text = self.current_node_info.get("text", "")
        return self
