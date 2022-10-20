from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Reddit
import torch
from torch_geometric.loader import NeighborSampler
# from torch_geometric.loader. import Adj, EdgeIndex
# from NS import NeighborSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from SAGEConv import SAGEConv
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import time
from typing import Callable, List, NamedTuple, Optional, Tuple, Union
from torch import Tensor
from torch_sparse import SparseTensor
# import numpy as np

class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)

class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)

class MyNeighborSampler(NeighborSampler):
    def __init__(self, edge_index,
                 sizes, node_idx=None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True,
                 transform: Callable = None,
                  **kwargs):
        super().__init__(edge_index, sizes, node_idx, num_nodes, return_e_id,
                            transform, **kwargs)
        self._generate_randSeq([25,10], [10000,10000])
        # edge_index = edge_index.to('cpu')

        # if 'collate_fn' in kwargs:
        #     del kwargs['collate_fn']
        # if 'dataset' in kwargs:
        #     del kwargs['dataset']

        # # Save for Pytorch Lightning < 1.6:
        # self.edge_index = edge_index
        # self.node_idx = node_idx
        # self.num_nodes = num_nodes

        # self.sizes = sizes
        # self.return_e_id = return_e_id
        # self.transform = transform
        # self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        # self.__val__ = None

        # # Obtain a *transposed* `SparseTensor` instance.
        # if not self.is_sparse_tensor:
        #     if (num_nodes is None and node_idx is not None
        #             and node_idx.dtype == torch.bool):
        #         num_nodes = node_idx.size(0)
        #     if (num_nodes is None and node_idx is not None
        #             and node_idx.dtype == torch.long):
        #         num_nodes = max(int(edge_index.max()), int(node_idx.max())) + 1
        #     if num_nodes is None:
        #         num_nodes = int(edge_index.max()) + 1

        #     value = torch.arange(edge_index.size(1)) if return_e_id else None
        #     self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
        #                               value=value,
        #                               sparse_sizes=(num_nodes, num_nodes)).t()
        # else:
        #     adj_t = edge_index
        #     if return_e_id:
        #         self.__val__ = adj_t.storage.value()
        #         value = torch.arange(adj_t.nnz())
        #         adj_t = adj_t.set_value(value, layout='coo')
        #     self.adj_t = adj_t

        # self.adj_t.storage.rowptr()

        # if node_idx is None:
        #     node_idx = torch.arange(self.adj_t.sparse_size(0))
        # elif node_idx.dtype == torch.bool:
        #     node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        # super().__init__(
        #     node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def _generate_randSeq(self, sizes: List[int], num_seq: List[int]):
        rand_mat = []
        print("generate sample patterns")
        for i,size in enumerate(sizes):
            num = num_seq[i]
            res = np.random.randint(low=0, high=1e10, size=(num, size)).tolist()
            rand_mat.append(res)
        self.rand_mat = rand_mat
        self.num_seq = num_seq

    def sample_adj(self, rowptr, col, idx, num_neighbors, layer_depth,replace):
        rowptr, col, idx = rowptr.tolist(), col.tolist(), idx.tolist()
        out_rowptr = [0] * (len(idx) + 1)
        out_rowptr[0] = 0
        n_ids = []
        n_id_map = {}
        cols = [] # col, e_id

        for n,i in enumerate(idx):
            cols.append(list())
            n_id_map[i] = n
            n_ids.append(i)
        
        if num_neighbors < 0:
            for i in range(len(idx)):
                n = idx[i]
                row_start, row_end = rowptr[n], rowptr[n+1]
                row_count = row_end - row_start
                if row_count > 0:
                    for j in range(num_neighbors):
                        e = row_start  + j
                        c = col[e]

                        if n_id_map.get(c) is None:
                            n_id_map[c] = len(n_ids)
                            n_ids.append(c)
                        cols[i].append((n_id_map[c], e))
                out_rowptr[i+1] = out_rowptr[i] + len(cols[i])
        elif replace:
            for i in range(len(idx)):
                n = idx[i]
                row_start, row_end = rowptr[n], rowptr[n+1]
                row_count = row_end - row_start
                if row_count > 0:
                    # generate start index
                    path_id =np.random.randint(0, self.num_seq[layer_depth])
                    for j in range(num_neighbors):
                        # e = row_start  + (start + self.rand_mat[layer_depth][path_id][j]) % row_count
                        start = np.random.randint(low=0, high=row_count)
                        e = row_start + start
                        c = col[e]
                        if n_id_map.get(c) is None:
                            n_id_map[c] = len(n_ids)
                            n_ids.append(c)
                        cols[i].append((n_id_map[c], e))
                out_rowptr[i+1] = out_rowptr[i] + len(cols[i])
        else:
            pass
        
        out_e_id = []
        out_col = []

        # sort
        for col_vec in cols:
            col_vec.sort(key = lambda x : x[0])
            for col_data, e_id_data in col_vec:
                out_col.append(col_data)
                out_e_id.append(e_id_data)                
            
        out_rowptr = torch.LongTensor(out_rowptr)
        out_col = torch.LongTensor(out_col)
        out_n_id = torch.LongTensor(n_ids)
        out_e_id = torch.LongTensor(out_e_id)
        return out_rowptr, out_col, out_n_id, out_e_id

    def sample_helper(self, subset, num_neighbors, layer_depth, replace=True):
        # if not self.rand_mat:
            # print("generate sample pattern")
            
        rowptr, col, value = self.adj_t.csr()

        # rowptr, col, n_id, e_id = torch.ops.torch_sparse.sample_adj(
        #     rowptr, col, subset, num_neighbors, replace)
        rowptr, col, n_id, e_id = self.sample_adj(rowptr, col, subset, num_neighbors, layer_depth, replace)

        if value is not None:
            value = value[e_id]

        out = SparseTensor(rowptr=rowptr, row=None, col=col, value=value,
                        sparse_sizes=(subset.size(0), n_id.size(0)),
                        is_sorted=True)

        return out, n_id
        
    
    def sample(self, batch):
        # print("running customized sampler...")
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for depth, size in enumerate(self.sizes): # self.sizes is the sample number for each layer
            # adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=True)
            # rewrite the sample method.
            adj_t, n_id = self.sample_helper(n_id, size, depth,replace=True)

            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            if self.__val__ is not None:
                adj_t.set_value_(self.__val__[e_id], layout='coo')

            if self.is_sparse_tensor:
                adjs.append(Adj(adj_t, e_id, size))
            else:
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([col, row], dim=0)
                adjs.append(EdgeIndex(edge_index, e_id, size))

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (batch_size, n_id, adjs)
        out = self.transform(*out) if self.transform is not None else out
        return out
        # pass
# from torch.utils.data import DataLoader

"""
GraphSAGE的minibatch方法(包含采样)
可选择的数据集：Cora、Citeseer、Pubmed、Reddit
使用方法：GS方法
"""

# dataset = Planetoid(root='./cora/', name='Cora')
# dataset = Planetoid(root='./citeseer',name='Citeseer')
# dataset = Planetoid(root='./pubmed/',name='Pubmed')
dataset = Reddit(root='./reddit/')
print(dataset)

start_time = time.time()
# cora
# train_loader = NeighborSampler(dataset[0].edge_index, node_idx=dataset[0].train_mask,
#                                sizes=[10, 10], batch_size=16, shuffle=True,
#                                num_workers=12)

# Reddit
train_loader = MyNeighborSampler(dataset[0].edge_index, node_idx=dataset[0].train_mask,
                               sizes=[25, 10], batch_size=1024, shuffle=True,
                               num_workers=12)

end_time = time.time()
init_sample_time = end_time - start_time
# print('NeighborSampler time:{}'.format(end_time - start_time))

subgraph_loader = MyNeighborSampler(dataset[0].edge_index, node_idx=None, sizes=[-1],
                                  batch_size=1024, shuffle=False,
                                  num_workers=12)


class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGENet, self).__init__()

        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.

        lin_times = 0
        mes_times = 0
        aggr_times = 0
        up_times = 0

        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x, linear_time, message_time, aggregate_time, update_time = self.convs[i]((x, x_target), edge_index)
            lin_times += linear_time
            mes_times += message_time
            aggr_times += aggregate_time
            up_times += update_time
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1), lin_times, mes_times, aggr_times, up_times

    def inference(self, x_all):
        # pbar = tqdm(total=x_all.size(0) * self.num_layers)
        # pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x, linear_time, message_time, aggregate_time, update_time = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                # pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        # pbar.close()

        return x_all


# cora
# model = SAGENet(dataset.num_features, 16, dataset.num_classes)

# Reddit
model = SAGENet(dataset.num_features, 256, dataset.num_classes)
print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
data = dataset[0].to(device)
print(data)

x = data.x.to(device)
y = data.y.squeeze().to(device)

criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train(epoch):
    model.train()

    # pbar = tqdm(total=int(data.train_mask.sum()))
    # pbar.set_description(f'Epoch {epoch:02d}')

    total_lin_time = 0
    total_mes_time = 0
    total_aggr_time = 0
    total_up_time = 0

    total_sample_time = 0

    total_loss = total_correct = 0
    start_time = time.time()
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        end_time = time.time()
        total_sample_time += (end_time - start_time)

        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out, lin_time, mes_time, aggr_time, up_time = model(x[n_id], adjs)

        total_lin_time += lin_time
        total_mes_time += mes_time
        total_aggr_time += aggr_time
        total_up_time += up_time

        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        # pbar.update(batch_size)
        start_time = time.time()

    # pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc, total_lin_time, total_mes_time, total_aggr_time, total_up_time, total_sample_time


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results


lin_times = []
mes_times = []
aggr_times = []
up_times = []
sample_times = []
for epoch in range(1, 11):
    loss, acc, lin_time, mes_time, aggr_time, up_time, sample_time = train(epoch)

    lin_times.append(lin_time)
    mes_times.append(mes_time)
    aggr_times.append(aggr_time)
    up_times.append(up_time)
    sample_times.append(sample_time)

    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')

    train_acc, val_acc, test_acc = test()
    print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')

print("Average linear time:", 1000 * np.mean(lin_times), 'ms')
print("Average message time:", 1000 * np.mean(mes_times), 'ms')
print("Average aggregate time:", 1000 * np.mean(aggr_times), 'ms')
print("Average update time:", 1000 * np.mean(up_times), 'ms')
print("Average sample time:", (1000 * np.mean(sample_times) + init_sample_time), 'ms')
