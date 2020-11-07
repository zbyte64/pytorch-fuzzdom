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


DomInfo = namedtuple(
    "DomInfo",
    [
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
        "row",
        "ref",
        "col",
        "left",
    ],
)


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
        assert len(dom_info.ref)
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
        index = self.dom_info.ref.index(ref)
        self.clipboard_text = self.dom_info.text[index]
        return self
