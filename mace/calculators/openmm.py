from e3nn.util import jit
import torch
from torch_nl import compute_neighborlist
import mace
from mace.calculators.neighbour_list_torch import primitive_neighbor_list_torch
from mace import data
from mace.tools import torch_geometric, utils
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory
import openmm
from typing import Optional, Iterable


torch.set_default_dtype(torch.float64)


def compile_model(model_path):
    model = torch.load(model_path)
    res = {}
    res["model"] = jit.compile(model)
    res["z_table"] = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    res["r_max"] = model.r_max
    return res


class MACE_openmm(torch.nn.Module):
    def __init__(self, model_path, atoms_obj):
        super().__init__()
        dat = compile_model(model_path)
        config = data.config_from_atoms(atoms_obj)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=dat["z_table"], cutoff=dat["r_max"]
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch_dict = next(iter(data_loader)).to_dict()
        batch_dict.pop("edge_index")
        batch_dict.pop("energy", None)
        batch_dict.pop("forces", None)
        batch_dict.pop("positions")
        # batch_dict.pop("shifts")
        batch_dict.pop("weight")
        self.inp_dict = batch_dict
        self.model = dat["model"]
        self.r_max = dat["r_max"]

    def forward(self, positions):
        sender, receiver, unit_shifts = primitive_neighbor_list_torch(
            quantities="ijS",
            pbc=(False, False, False),
            cell=self.inp_dict["cell"],
            positions=positions,
            cutoff=self.r_max,
            self_interaction=True,  # we want edges from atom to itself in different periodic images
            use_scaled_positions=False,  # positions are not scaled positions
            device="cpu",
        )
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= torch.all(unit_shifts == 0, dim=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]
        # Build output
        edge_index = torch.stack((sender, receiver))  # [2, n_edges]

        # From the docs: With the shift vector S, the distances D between atoms can be computed from
        # D = positions[j]-positions[i]+S.dot(cell)
        # shifts = torch.dot(unit_shifts, self.inp_dict["cell"])  # [n_edges, 3]
        inp_dict_this_config = self.inp_dict.copy()
        inp_dict_this_config["positions"] = positions
        inp_dict_this_config["edge_index"] = edge_index
        # inp_dict_this_config["shifts"] = shifts
        # inp_dict_this_config[""] =
        res = self.model(inp_dict_this_config)
        return (res["energy"], res["forces"])


class MACE_openmm2(torch.nn.Module):
    def __init__(self, model_path, atoms_obj, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        dat = compile_model(model_path)
        config = data.config_from_atoms(atoms_obj)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=dat["z_table"], cutoff=dat["r_max"]
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)
        batch_dict = batch.to_dict()
        batch_dict.pop("edge_index")
        batch_dict.pop("energy", None)
        batch_dict.pop("forces", None)
        batch_dict.pop("positions")
        # batch_dict.pop("shifts")
        batch_dict.pop("weight")
        self.inp_dict = batch_dict
        self.model = dat["model"]
        self.r_max = dat["r_max"]

    def forward(self, positions):
        bbatch = torch.zeros(positions.shape[0], dtype=torch.long, device=self.device)
        mapping, batch_mapping, shifts_idx = compute_neighborlist(
            self.r_max,
            positions,
            self.inp_dict["cell"],
            torch.tensor([False, False, False], device=self.device),
            bbatch,
            self_interaction=True,
        )

        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = mapping[0] == mapping[1]
        true_self_edge &= torch.all(shifts_idx == 0, dim=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = mapping[0][keep_edge]
        receiver = mapping[1][keep_edge]
        shifts_idx = shifts_idx[keep_edge]

        edge_index = torch.stack((sender, receiver))

        # From the docs: With the shift vector S, the distances D between atoms can be computed from
        # D = positions[j]-positions[i]+S.dot(cell)
        # shifts = torch.dot(unit_shifts, self.inp_dict["cell"])  # [n_edges, 3]
        inp_dict_this_config = self.inp_dict.copy()
        inp_dict_this_config["positions"] = positions
        inp_dict_this_config["edge_index"] = edge_index
        inp_dict_this_config["shifts"] = shifts_idx

        # inp_dict_this_config["shifts"] = shifts
        # inp_dict_this_config[""] =
        res = self.model(inp_dict_this_config)
        return (res["energy"], res["forces"])



class MACEPotentialImplFactory(MLPotentialImplFactory):
    """
    A factory to create ACEPotential objects.
    """

    def createImpl(self, name: str, **args) -> MLPotentialImpl:
        return MACEPotentialImpl(name, **args)


class MACEPotentialImpl(MLPotentialImpl):
    """
    This is the MLPotentialImpl implementing the MACE model built using the pytorch version of MACE.


    MACE does not have a central location to distribute models yet and so we assume the model is in a local
    file and must be passed to the init.
    """

    def __init__(self, name: str, model_path: str):
        self.name = name
        self.model_path = model_path

    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  filename: str = "macemodel.pt",
                  **args):
        """Create the mace model"""
        import torch
        import openmmtorch
        from mace.data.utils import Configuration, AtomicNumberTable
        from mace.tools.torch_geometric import DataLoader
        from mace.data import AtomicData
        from e3nn.util import jit
        import numpy as np
        from torch_nl import compute_neighborlist

        # load the mace model
        model = torch.load(self.model_path)
        z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
        # find the atomic numbers
        target_atoms = list(topology.atoms())
        if atoms is not None:
            target_atoms = [target_atoms[i] for i in atoms]
        atomic_numbers = np.array([atom.element.atomic_number for atom in target_atoms])
        is_periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()

        if is_periodic:
            pbc = (True, True, True)
        else:
            pbc = (False, False, False)

        config = Configuration(
            atomic_numbers=atomic_numbers,
            positions=np.zeros((len(atomic_numbers), 3)),
            pbc=pbc,
            cell=np.zeros((3, 3)),
            weight=1
        )
        data_loader = DataLoader(dataset=[AtomicData.from_config(config, z_table=z_table, cutoff=model.r_max)],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        input_dict = next(iter(data_loader)).to_dict()
        input_dict.pop("edge_index")
        input_dict.pop("energy", None)
        input_dict.pop("forces", None)
        input_dict.pop("positions")
        input_dict.pop("shifts")
        input_dict.pop("weight")


        class MACEForce(torch.nn.Module):
            """A wrapper around a MACe model which can be called by openmm-torch"""

            def __init__(self, model, input_data, atoms, periodic):
                """
                Args:
                    model:
                        The MACE model which should be wrapped by the class
                    node_attrs:
                        A tensor of the one hot encoded atom nodes
                    atoms:
                        The indices of the atoms in the openmm topology object, used to extract the correct positions
                    periodic:
                        If the system cell is periodic or not, used to calculate the atomic distances
                """

                super(MACEForce, self).__init__()
                self.model = model
                self.input_dict = input_data
                if atoms is None:
                    self.indices = None
                else:
                    self.indices = torch.tensor(sorted(atoms), dtype=torch.int64)
                if periodic:
                    self.pbc = torch.tensor([True, True, True], dtype=torch.bool, requires_grad=False)
                else:
                    self.pbc = torch.tensor([False, False, False], dtype=torch.bool, requires_grad=False)

            def forward(self, positions, boxvectors: Optional[torch.Tensor] = None):
                """
                Evaluate the mace model on the selected atoms.

                Args:
                    positions: torch.Tensor shape (nparticles, 3)
                        The positions of all atoms in the system in nanometers.
                    boxvectors: torch.Tensor shape (3,3)
                        The box vectors of the periodic cell in nanometers

                Returns:
                    energy: torch.Scalar
                        The potential energy in KJ/mol
                    forces: torch.Tensor shape (nparticles, 3)
                        The forces on each atom in (KJ/mol/nm)
                """
                # MACE expects inputs in Angstroms
                # create a config for the model
                positions = positions.to(torch.float64)
                # if we are only modeling a subsection select those positions
                if self.indices is not None:
                    positions = positions[self.indices]
                positions = 10.0*positions

                if boxvectors is None:
                    cell = torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]], requires_grad=False, dtype=torch.float64)
                else:
                    boxvectors = boxvectors.to(torch.float64)
                    cell = 10.0*boxvectors

                # pass through the model
                mapping, batch_mapping, shifts_idx = compute_neighborlist(
                    cutoff=self.model.r_max,
                    pos=positions,
                    cell=cell,
                    pbc=self.pbc,
                    batch=torch.zeros(positions.shape[0], dtype=torch.long),
                    self_interaction=True)

                # Eliminate self-edges that don't cross periodic boundaries
                true_self_edge = mapping[0] == mapping[1]
                true_self_edge &= torch.all(shifts_idx == 0, dim=1)
                keep_edge = ~true_self_edge

                # Note: after eliminating self-edges, it can be that no edges remain in this system
                sender = mapping[0][keep_edge]
                receiver = mapping[1][keep_edge]
                shifts_idx = shifts_idx[keep_edge]

                edge_index = torch.stack((sender, receiver))
                inp_dict_this_config = self.input_dict.copy()
                inp_dict_this_config["positions"] = positions
                inp_dict_this_config["edge_index"] = edge_index
                inp_dict_this_config["shifts"] = shifts_idx
                inp_dict_this_config["cell"] = cell

                res = self.model(inp_dict_this_config, compute_force=True)
                return 96.486*res["energy"]


        mace_force = MACEForce(model=model, input_data=input_dict, atoms=atoms, periodic=is_periodic)

        # convert using jit
        module = jit.script(mace_force)
        module.save(filename)

        # Create the openmm torch force

        force = openmmtorch.TorchForce(filename)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(is_periodic)
        # force.setOutputsForces(True)
        system.addForce(force)


MLPotential.registerImplFactory("mace", MACEPotentialImplFactory())