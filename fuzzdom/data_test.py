import pytest
import os
import logging
import networkx as nx
import torch
from torch_geometric.data import Batch, Data
from .data import vectorize_projections

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def test_project_vectors():
    """
    Test index broadcasts from different orientations
    """
    g = nx.DiGraph()
    for i in range(3):
        g.add_node(i, index=1)
        if i:
            g.add_edge(i - 1, i)
    v = vectorize_projections(
        {
            "ux_action": [{"action_idx": i + 10} for i in range(5)],
            "field": [{"field_idx": i + 1} for i in range(2)],
            "leaf": [{"dom_idx": i} for i in range(7)],
        },
        g,
        source_domain="dom",
        final_domain="action",
        add_intersections=[("field", "ux_action"), ("ux_action", "field")],
    )
    field_actions = torch.ger(torch.arange(1, 3), torch.arange(10, 15))
    p = torch.cat(
        [
            field_actions.view(-1, 1)[v.field_ux_action_index],
            v.action_idx.view(-1, 1),
            v.field_idx.view(-1, 1),
        ],
        dim=1,
    )
    m = p[:, 0] == p[:, 1] * p[:, 2]
    assert m.min().item(), str(p)

    action_fields = torch.ger(torch.arange(10, 15), torch.arange(1, 3))
    p = torch.cat(
        [
            field_actions.view(-1, 1)[v.ux_action_field_index],
            v.action_idx.view(-1, 1),
            v.field_idx.view(-1, 1),
        ],
        dim=1,
    )
    m = p[:, 0] == p[:, 1] * p[:, 2]
    assert m.min().item(), str(p)


def test_project_vectors_batch_packing():
    g = nx.DiGraph()
    for i in range(3):
        g.add_node(i, index=1)
        if i:
            g.add_edge(i - 1, i)
    v1 = vectorize_projections(
        {
            "ux_action": [{"action_idx": i + 10} for i in range(5)],
            "field": [{"field_idx": i + 1} for i in range(2)],
            "leaf": [{"dom_idx": i} for i in range(7)],
        },
        g,
        source_domain="dom",
        final_domain="action",
        add_intersections=[("ux_action", "field")],
    )
    field_actions1 = torch.ger(torch.arange(1, 3), torch.arange(10, 15))

    for i in range(3, 5):
        g.add_node(i, index=1)
        g.add_edge(i - 1, i)
    v2 = vectorize_projections(
        {
            "ux_action": [{"action_idx": i + 10} for i in range(5)],
            "field": [{"field_idx": i + 1} for i in range(3)],
            "leaf": [{"dom_idx": i} for i in range(7)],
        },
        g,
        source_domain="dom",
        final_domain="action",
        add_intersections=[("ux_action", "field")],
    )
    field_actions2 = torch.ger(torch.arange(1, 4), torch.arange(10, 15))
    batch = Batch.from_data_list([v1, v2])

    a = torch.cat([field_actions1, field_actions2], dim=0)
    p = torch.cat(
        [
            a.view(-1, 1)[batch.ux_action_field_index],
            batch.action_idx.view(-1, 1),
            batch.field_idx.view(-1, 1),
        ],
        dim=1,
    )
    m = p[:, 0] == p[:, 1] * p[:, 2]
    assert m.min().item(), str(p)
