# Finetuned Models for Inference
import json
from pathlib import Path

import smirk
from smirk import SmirkTokenizerFast  # noqa: F401
from smirk import SmirkTokenizerFast as SmirkTokenizer  # noqa: F401
import torch
from safetensors.torch import save_model, load_model
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from .normalize import AbstractNormalizer


class PredictionTaskHead(nn.Module):
    def __init__(
        self, embed_dim: int, output_size: int = 1, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.desc_skip_connection = True

        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.GELU()
        self.final = nn.Linear(embed_dim, output_size)

    def forward(self, emb):
        emb = emb[:, 0, :]
        x_out = self.fc1(emb)
        x_out = self.dropout1(x_out)
        x_out = self.relu1(x_out)

        if self.desc_skip_connection is True:
            x_out = x_out + emb

        z = self.fc2(x_out)
        z = self.dropout2(z)
        z = self.relu2(z)
        if self.desc_skip_connection is True:
            z = self.final(z + x_out)
        else:
            z = self.final(z)
        return z


class MISTFinetuned(torch.nn.Module):
    def __init__(
        self,
        encoder,
        task_network,
        transform,
        tokenizer,
        channels: list[str] | list[dict[str, str]] | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.task_network = task_network
        self.transform = transform
        self.tokenizer = tokenizer
        if channels is not None:
            channels = list(
                {"name": chn} if isinstance(chn, str) else chn
                for chn in channels
            )
        self.channels = channels

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hs = self.encoder(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state
        y = self.task_network(hs)
        return self.transform.forward(y)

    def save_pretrained(self, save_directory):
        config = {
            "encoder": self.encoder.config.to_diff_dict(),
            "task_network": {
                "embed_dim": self.encoder.config.hidden_size,
                "output_size": self.task_network.final.out_features,
                "dropout": self.task_network.dropout1.p,
            },
            "transform": self.transform.to_config(),
            "channels": self.channels,
        }

        Path(save_directory, "config.json").write_text(
            json.dumps(config, indent=4)
        )
        save_model(self, Path(save_directory, "model.safetensors"))

    def predict(self, smi: list[str]):
        batch = self.tokenizer(smi)
        collate_fn = DataCollatorWithPadding(self.tokenizer)
        device = next(self.encoder.parameters()).get_device()
        batch = collate_fn(batch)
        batch = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        out = self(**batch)
        if self.channels is None:
            return out

        out_batch = []
        for idx in range(len(smi)):
            out_batch.append(self.annotate_prediction(out[idx, :].cpu()))

        return out_batch

    @classmethod
    def from_pretrained(self, save_directory: str):
        config = json.loads(Path(save_directory, "config.json").read_text())
        encoder_config = AutoConfig.for_model(
            config["encoder"]["model_type"]
        ).from_dict(config["encoder"])
        encoder = AutoModel.from_config(encoder_config, add_pooling_layer=False)
        task_network = PredictionTaskHead(**config["task_network"])
        transform_config = config["transform"]
        if "class" in transform_config:
            transform_config["transform"] = transform_config.pop("class")
        transform = AbstractNormalizer.get(**transform_config)

        # Instantiate model
        tokenizer = AutoTokenizer.from_pretrained(save_directory)
        model = MISTFinetuned(
            encoder,
            task_network,
            transform,
            tokenizer=tokenizer,
            channels=config["channels"],
        )

        load_model(model, Path(save_directory).joinpath("model.safetensors"))
        return model
