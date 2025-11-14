from typing import Annotated

from openbabel import openbabel
from pydantic import AfterValidator, BaseModel


def validate_and_canonicalize_smiles(smi: str):
    mol = openbabel.OBMol()
    conv = openbabel.OBConversion()
    conv.SetInFormat("smi")

    # Try to parse input SMILES
    if not conv.ReadString(mol, smi) or mol.NumAtoms() == 0:
        raise ValueError(f"Invalid SMILES string: {smi}")

    # Convert back to canonical OpenSMILES
    conv.SetOutFormat("smi")
    conv.AddOption("c", conv.OUTOPTIONS)  # 'c' = canonical
    canonical_smiles = conv.WriteString(mol).strip()

    # Sometimes trailing newlines are added
    if not canonical_smiles:
        raise ValueError(f"Failed to generate canonical SMILES from: {smi}")

    return canonical_smiles


SMILES = Annotated[str, AfterValidator(validate_and_canonicalize_smiles)]


class Property(BaseModel):
    value: float | int
    units: str | None


class MolecularProperties(BaseModel):
    molecule: SMILES
    properties: dict[str, Property]
