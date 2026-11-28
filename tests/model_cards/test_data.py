import json

import numpy as np
import pytest

from ursa.model_cards.data import inspect_data

h5py = pytest.importorskip("h5py")


def test_smiles_list_and_table_with_targets(tmp_path):
    smi = tmp_path / "mols.smi"
    smi.write_text(
        "SMILES\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C caffeine\nCCO ethanol\n"
    )
    profile = inspect_data(smi)
    assert (profile.kind, profile.representation, profile.n_samples) == (
        "molecule",
        "SMILES",
        2,
    )
    assert profile.examples[0].startswith("CN1C")

    csv = tmp_path / "props.csv"
    csv.write_text(
        "id,smiles,solubility\n1,CCO,0.5\n2,c1ccccc1,-1.6\n3,CC(=O)O,1.2\n"
    )
    table = inspect_data(csv)
    assert table.kind == "molecule" and table.n_samples == 3
    # The id column is not a target; the solubility column is.
    assert table.labels == ["solubility"]


def test_fasta_protein_sequences(tmp_path):
    fasta = tmp_path / "prot.fasta"
    fasta.write_text(
        ">a\nMKTAYIAKQRQISFVKSHFSRQ\n>b\nMVLSPADKTNVKAAWGKVGAHAG\n"
    )
    profile = inspect_data(fasta)
    assert profile.representation == "protein sequence"
    assert profile.n_samples == 2


def test_netcdf3_grid_resolution_and_time_step(tmp_path):
    from scipy.io import netcdf_file

    path = tmp_path / "era.nc"
    with netcdf_file(str(path), "w") as handle:
        handle.createDimension("time", 3)
        handle.createDimension("lat", 181)
        handle.createDimension("lon", 360)
        time = handle.createVariable("time", "f8", ("time",))
        time[:] = [0, 6, 12]
        time.units = "hours since 2020-01-01"
        lat = handle.createVariable("lat", "f4", ("lat",))
        lat[:] = np.linspace(90, -90, 181)
        lon = handle.createVariable("lon", "f4", ("lon",))
        lon[:] = np.arange(360.0)
        t2m = handle.createVariable("t2m", "f4", ("time", "lat", "lon"))
        t2m[:] = 280.0
        t2m.units = "K"
    profile = inspect_data(path)
    assert profile.format == "NetCDF3" and profile.kind == "field"
    assert profile.grid == [181, 360] and profile.resolution == "1 deg"
    assert (profile.time_steps, profile.time_step) == (3, "6 hours")
    assert profile.variables[0].units == "K"


def test_netcdf4_style_hdf5_with_dimension_scales(tmp_path):
    path = tmp_path / "pl.nc"
    with h5py.File(path, "w") as handle:
        for name, values in {
            "level": [500.0, 850.0],
            "lat": np.linspace(90, -90, 5),
            "lon": np.arange(0, 360, 90.0),
        }.items():
            handle[name] = np.asarray(values)
            handle[name].make_scale(name)
        handle["z"] = np.zeros((2, 5, 4), dtype="f4")
        for axis, name in enumerate(["level", "lat", "lon"]):
            handle["z"].dims[axis].attach_scale(handle[name])
    profile = inspect_data(path)
    assert profile.format == "NetCDF4" and profile.spatial_dims == 2
    assert profile.resolution == "45 x 90 deg"
    (z,) = profile.variables
    assert z.dims == ["level", "lat", "lon"] and z.levels == [500.0, 850.0]


def test_the_well_hdf5_layout(tmp_path):
    path = tmp_path / "state.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["dataset_name"] = "heat_ic"
        handle.attrs["n_spatial_dims"] = 2
        handle.attrs["grid_type"] = "cartesian"
        handle.create_group("t0_fields")["av_density"] = np.ones((1, 1, 8, 4))
    profile = inspect_data(path)
    assert profile.representation == "The Well HDF5"
    assert profile.spatial_dims == 2 and profile.grid == [8, 4]
    assert [v.name for v in profile.variables] == ["av_density"]
    assert "; 1 sample;" in profile.summary()


def test_arrays_structures_and_json(tmp_path):
    np.savez(tmp_path / "train.npz", x=np.zeros((4, 16, 16)), y=np.arange(4.0))
    arrays = inspect_data(tmp_path / "train.npz")
    assert arrays.kind == "tensor" and arrays.labels == ["y"]

    (tmp_path / "w.xyz").write_text(
        "3\nwater\nO 0 0 0\nH 0 0 1\nH 0 1 0\n", encoding="utf-8"
    )
    xyz = inspect_data(tmp_path / "w.xyz")
    assert xyz.kind == "structure" and xyz.n_samples == 1
    assert "Elements: H, O." in xyz.notes

    (tmp_path / "s.cif").write_text(
        "data_x\n_chemical_formula_sum 'Na Cl'\n", encoding="utf-8"
    )
    assert "Formula: Na Cl." in inspect_data(tmp_path / "s.cif").notes

    (tmp_path / "m.json").write_text(
        json.dumps(["CCO", "c1ccccc1"]), encoding="utf-8"
    )
    assert inspect_data(tmp_path / "m.json").representation == "SMILES"


def test_missing_and_unsupported_files_explain_themselves(tmp_path):
    assert inspect_data(tmp_path / "nope.nc").notes == [
        "The file does not exist."
    ]
    odd = tmp_path / "data.xyzzy"
    odd.write_text("?", encoding="utf-8")
    assert "not supported yet" in inspect_data(odd).notes[0]
