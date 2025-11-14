from pathlib import Path
from threading import Lock
from textwrap import dedent

import torch
from pydantic import Field, BaseModel
from transformers import DataCollatorWithPadding

from ursa.tools.fm_base_tool import TorchModuleTool

from .smiles import SMILES, MolecularProperties, Property
from .vendor.mist.model import MISTFinetuned


class Molecule(BaseModel):
    smi: SMILES


class MistModel(TorchModuleTool):
    channels: list[str]
    tokenizer_lock: Lock = Field(default_factory=lambda: Lock())

    args_schema: type[BaseModel] = Field(default=Molecule)
    output_schema: type[BaseModel] = Field(default=MolecularProperties)

    @classmethod
    def get_description(cls, channels: list[dict] | None):
        header = """\
        Predicts chemical properties for an input molecule encoded with SMILES.

        Properties Predicted:
        """
        lines = [dedent(header)]
        for chn in channels:
            name = chn["name"]
            description = chn.get("description")
            units = chn.get("units")
            match units, description:
                case None, None:
                    lines.append(f"- {name}")
                case str(), None:
                    lines.append(f"- {name} in units of {units}")
                case None, str():
                    lines.append(f"- {name}: {description}")
                case str(), str():
                    lines.append(
                        f"- {chn['name']}: {chn['description']} in units of {chn['units']}"
                    )

        return "\n".join(lines)

    @classmethod
    def from_pretrained(
        cls,
        save_directory: str | Path,
        channels: list[str] | None = None,
        batch_size: int = 32,
    ):
        save_directory = Path(save_directory)
        model = MISTFinetuned.from_pretrained(save_directory)
        if device := torch.accelerator.current_accelerator():
            model = model.to(device)
        assert isinstance(model, MISTFinetuned)
        model.compile()

        return MistModel(
            name=save_directory.name.replace(".", "p"),
            description=cls.get_description(model.channels),
            channels=channels or [chn["name"] for chn in model.channels],
            fm=model,
            batch_size=batch_size,
            device=device,
        )

    def preprocess(self, molecules: list[Molecule]):
        with self.tokenizer_lock:
            collate_fn = DataCollatorWithPadding(self.fm.tokenizer)
            smis = [mol.smi for mol in molecules]
            tokens = collate_fn(self.fm.tokenizer(smis))
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        return {"molecules": molecules, **tokens}

    def _forward(self, model_input):
        molecules = model_input.pop("molecules")
        y = self.fm(**model_input).to("cpu")
        return {"molecules": molecules, "output": y}

    def postprocess(self, model_output):
        properties = {}
        y = model_output["output"]
        molecules = model_output["molecules"]
        for mol, y in zip(molecules, y):
            for v, chn in zip(y, self.fm.channels):
                if chn["name"] not in self.channels:
                    pass
                properties[chn["name"]] = Property(
                    value=v, units=chn.get("units", None)
                )
            yield MolecularProperties(molecule=mol.smi, properties=properties)
