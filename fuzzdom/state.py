from PIL import Image
from copy import copy
import networkx as nx
from collections import namedtuple

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


DomInfo = namedtuple("DomInfo", ["nodes", "edges"])


class MiniWoBGraphState(object):
    """MiniWoB Graph state.
    """

    def __init__(
        self,
        utterance: str,
        fields: Fields,
        dom_info: DomInfo,
        screenshot: Image,
        logs: dict,
    ):
        assert isinstance(dom_info, DomInfo)
        assert len(dom_info.nodes)
        self.utterance = utterance
        self.fields = fields
        self.dom_info = dom_info
        self.logs = logs
        if screenshot:
            add_image_slices_to_graph(self.dom_info, screenshot)
        self.screenshot = screenshot
        self.clipboard_text = ""

    def copy_node_text(self, ref):
        self = copy(self)
        node_info = next(filter(lambda x: x["ref"] == ref, self.dom_info.nodes))
        self.clipboard_text = node_info.get("text", "")
        return self
