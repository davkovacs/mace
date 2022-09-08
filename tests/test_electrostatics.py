import numpy as np
import torch

from mace import data, modules, tools
from mace.tools import torch_geometric
from mace.tools.scatter import scatter_sum


def test_FixedChargeDipoleBlock():
    print("Total Dipole moment test")
    X = torch.Tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    Q = torch.Tensor([-1.0, 1.0])
    MU = torch.Tensor([[0.0, 0.0, 0.0], [0.0, 1.0, -2.0]])

    config = data.utils.Configuration(
        atomic_numbers=np.array([8, 1]),
        positions=np.array(X),
        forces=np.array(
            [
                [0.0, -1.3, 0.0],
                [1.0, 0.2, 0.0],
            ]
        ),
        energy=-1.5,
        charges=np.array(Q),
        dipole=np.array(MU),
    )
    table = tools.AtomicNumberTable([1, 8])
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            atomic_data,
        ],
        batch_size=1,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    baseline = modules.FixedChargeDipoleBlock()
    tot_dipole = baseline(
        positions=batch.positions,
        charges=batch.charges,
        batch=batch.batch,
        num_graphs=batch.num_graphs,
    )
    assert torch.allclose(
        tot_dipole, torch.Tensor([1 / 0.208194, 1.0, -2.0]) - MU[0] - MU[1]
    )


def test_E_charge_mu():
    print("TEST CHARGE_DIPOLE ENERGY")
    X = torch.Tensor([[0.0, 0.0, 0.0], [-1.0, 2.0, -3.0]])
    Q = torch.Tensor([-2.0, 0.0])
    MU = torch.Tensor([[0.0, 0.0, 0.0], [-0.5, 2.0, 3.0]])
    MU = MU.repeat([3, 1])
    config = data.utils.Configuration(
        atomic_numbers=np.array([8, 1]),
        positions=np.array(X),
        forces=np.array(
            [
                [0.0, -1.3, 0.0],
                [1.0, 0.2, 0.0],
            ]
        ),
        energy=-1.5,
        charges=np.array(Q),
    )
    table = tools.AtomicNumberTable([1, 8])
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data, atomic_data],
        batch_size=3,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))

    softq_mu = modules.E_soft_q_mu(lambd=0.25, alpha=7.0)

    num_nodes = batch.node_attrs.shape[0]

    edge_index_long_r = modules.get_long_fully_connected_graph(
        batch=batch.batch, num_graphs=batch.num_graphs
    )
    sender, receiver = edge_index_long_r
    R_ij = batch.positions[sender] - batch.positions[receiver]  # [N_edges,3]
    d_ij = torch.pow(torch.norm(R_ij, dim=-1, keepdim=True), 2)
    R_ij_mu_j = torch.sum(R_ij * MU[sender], dim=-1, keepdim=True)  # [N_edges,1]
    E_q_mu_edges = softq_mu(
        d_ij, R_ij_mu_j, charge_i=batch.charges[receiver].unsqueeze(-1)
    )
    E_q_mu = scatter_sum(src=E_q_mu_edges, index=receiver, dim=0, dim_size=num_nodes)

    assert torch.allclose(torch.sum(E_q_mu) / 3, torch.Tensor([0.3551569692434432]))


def test_E_charge_charge():
    print("TEST CHARGE_CHARGE ENERGY")
    X = torch.Tensor([[0.0, 0.0, 0.0], [1.0, 2.0, -2.0]])
    Q = torch.Tensor([1.0, -2.0])

    config = data.utils.Configuration(
        atomic_numbers=np.array([8, 1]),
        positions=np.array(X),
        forces=np.array(
            [
                [0.0, -1.3, 0.0],
                [1.0, 0.2, 0.0],
            ]
        ),
        energy=-1.5,
        charges=np.array(Q),
    )
    table = tools.AtomicNumberTable([1, 8])
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))

    softq_q = modules.E_Gauss_qq(sigma=1.5)
    num_nodes = batch.node_attrs.shape[0]

    edge_index_long_r = modules.get_long_fully_connected_graph(
        batch=batch.batch, num_graphs=batch.num_graphs
    )
    sender, receiver = edge_index_long_r
    R_ij = batch.positions[sender] - batch.positions[receiver]  # [N_edges,3]
    d_ij = torch.norm(R_ij, dim=-1)  # [N_edges,1]
    q_i_q_j = batch.charges[receiver] * batch.charges[sender]  # [N_edges,1]
    E_q_q_edges = softq_q(d_ij, q_i_q_j)
    E_q_q = scatter_sum(src=E_q_q_edges, index=receiver, dim=0, dim_size=num_nodes)
    assert torch.allclose(torch.sum(E_q_q) / 2, torch.Tensor([-8.089728437505189]))


def test_E_dipole_dipole():
    print("TEST DIPOLE_DIPOLE ENERGY")
    X = torch.Tensor([[0.0, 0.0, 0.0], [-1.0, 2.0, -3.0]])
    MU = torch.Tensor([[1.0, 1.0, 0.0], [-1.0, 2.0, 3.0]])
    MU = MU.repeat([3, 1])
    config = data.utils.Configuration(
        atomic_numbers=np.array([8, 1]),
        positions=np.array(X),
        forces=np.array(
            [
                [0.0, -1.3, 0.0],
                [1.0, 0.2, 0.0],
            ]
        ),
        energy=-1.5,
    )
    table = tools.AtomicNumberTable([1, 8])
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data, atomic_data],
        batch_size=3,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))

    soft_mu_mu = modules.E_soft_mu_mu(lambd=0.25, alpha=7.0)
    num_nodes = batch.node_attrs.shape[0]

    edge_index_long_r = modules.get_long_fully_connected_graph(
        batch=batch.batch, num_graphs=batch.num_graphs
    )
    sender, receiver = edge_index_long_r
    R_ij = batch.positions[sender] - batch.positions[receiver]  # [N_edges,3]
    d_ij = torch.pow(torch.norm(R_ij, dim=-1, keepdim=True), 2)
    R_ij_mu_j = torch.sum(R_ij * MU[sender], dim=-1, keepdim=True)  # [N_edges,1]
    mu_i_R_ij = torch.sum(R_ij * MU[receiver], dim=-1, keepdim=True)  # [N_edges,1]
    mu_i_mu_j = torch.sum(
        MU[receiver] * MU[sender], dim=-1, keepdim=True
    )  # [N_edges,1]

    E_mu_mu_edges = soft_mu_mu(d_ij, mu_i_mu_j, R_ij_mu_j, mu_i_R_ij)
    E_mu_mu = scatter_sum(src=E_mu_mu_edges, index=receiver, dim=0, dim_size=num_nodes)

    assert torch.allclose(torch.sum(E_mu_mu) / 3, torch.Tensor([0.01371198570529033]))
