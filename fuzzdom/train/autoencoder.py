import os.path as osp

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import (
    train_test_split_edges,
    add_self_loops,
    remove_self_loops,
)

from fuzzdom.datasets import DomDataset
from fuzzdom.dir_paths import DATA_DIR
from fuzzdom.models import Encoder

torch.manual_seed(12345)

kwargs = {"GAE": GAE, "VGAE": VGAE}
dataset = DomDataset(DATA_DIR + "/dom-dataset", [])


class args:
    model = "GAE"
    dataset = "Cora"


print(dataset)
dataloader = dataset

channels = 8
dev = torch.device("cuda" if False and torch.cuda.is_available() else "cpu")
model = kwargs[args.model](Encoder(args.model, dataset.num_features, channels)).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(x, train_pos_edge_index):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    # print(z.shape, loss)  # , train_pos_edge_index)
    if args.model in ["VGAE"]:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def test(x, train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


for epoch in range(1, 51):
    for data in dataloader:
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data = train_test_split_edges(data)
        x, train_pos_edge_index = (
            data.x.to(dev),
            data.train_pos_edge_index.to(dev),
        )
        loss = train(x, train_pos_edge_index)
        print("loss", loss)
        # print("pos", train_pos_edge_index)
        # print("neg", data.test_neg_edge_index)
    auc, ap = test(
        x, train_pos_edge_index, data.test_pos_edge_index, data.test_neg_edge_index,
    )
    print("Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}".format(epoch, auc, ap))
torch.save(model, "./datadir/autoencoder.pt")
