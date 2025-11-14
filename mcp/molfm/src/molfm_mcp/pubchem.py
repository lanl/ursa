import logging

import pubchempy
from pydantic import BaseModel, ValidationError

from .smiles import SMILES


class PubChemMolecule(BaseModel):
    cid: int
    name: str
    iupac_name: str
    molecular_weight: float
    smi: SMILES


def search_pubchem(molecule: str):
    """Search for a molecule by name using PubChem"""
    out: list[PubChemMolecule] = []
    for compound in pubchempy.get_compounds(molecule, namespace="name"):
        try:
            mol = PubChemMolecule(
                cid=compound.cid,
                name=compound.synonyms[0],
                iupac_name=compound.iupac_name,
                molecular_weight=compound.molecular_weight,
                smi=compound.smiles,
            )
            out.append(mol)
        except ValidationError as e:
            logging.error(
                f"Failed to collate PubChemMolecule from {compound.to_dict()}: {e}",
            )

    return out if out else None
