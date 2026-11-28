"""Synthetic model cards for tests, one per common card dialect.

The models are fictional; the layouts follow the DOE Genesis template,
the Hugging Face template, the NVIDIA Model Card++ template, minimal
free-form READMEs, bare config files, Nomad MCP tool schemas, and the
URSA-native YAML card.
"""

GENESIS_CARD = """---
language:
- en
tags:
- project:genesis
- project:demo
- type:model
- science:physics-simulation
- science:fusion
- risk:general
license: apache-2.0
library_name: transformers
pipeline_tag: image-to-image
---

# PlasmaEdge-2D

PlasmaEdge-2D is a convolutional surrogate that predicts the evolution of edge-plasma density and temperature fields in a tokamak.

## Model short description

U-Net surrogate for 2D edge-plasma turbulence rollout.

## Model description

The model takes the plasma state at one time step as input and predicts the state one step later.

## Inputs and outputs

- Input fields: `density`, `electron_temperature`, `potential`, provided as a NumPy array of shape `(3, 256, 512)` in `float32`.
- Output fields: predicted `density`, `electron_temperature`, `potential` with the same shape.
- Normalization: each input channel is normalized to zero mean and unit variance with statistics from `stats.json`.
- Time step: 10 microseconds per rollout step.

## Compute Infrastructure

### Hardware

Inference requires a GPU with at least 12 GB of memory. Training used 4 NVIDIA A100 GPUs.

# Intended Uses

## Intended Use

Fast exploratory rollouts of edge-plasma turbulence for design studies.

## Out-of-Scope Use Cases

Operational plasma control decisions.

# How to use

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("example-org/plasmaedge-2d", trust_remote_code=True)
```

# Limitations

## Limitations

- Trained only on L-mode discharges.
- Accuracy degrades after 50 rollout steps.
"""

HF_CARD = """---
license: mit
library_name: pytorch
tags:
- materials
- crystal
datasets:
- example/mp-2024
metrics:
- mae
model-index:
- name: CrystalProp-S
  results:
  - task:
      type: regression
    dataset:
      name: MP band gaps
      type: example/mp-2024
      split: test
    metrics:
    - type: mae
      value: 0.21
      name: Band gap MAE
---

# CrystalProp-S: Crystal property prediction

## Model Details

### Model Description

CrystalProp-S is a graph neural network that predicts band gaps and formation energies of inorganic crystals.

- **Developed by:** Example Materials Lab
- **Model type:** Equivariant graph neural network
- **License:** MIT

## Uses

### Direct Use

Screening candidate crystals before DFT calculations.

## How to Get Started with the Model

```python
from crystalprop import CrystalProp
model = CrystalProp.from_pretrained("example/crystalprop-s")
```

## Technical Specifications

The model has 4.2M parameters and can run on CPU or GPU.

### Inputs

- **Inputs**: crystal structures as CIF files or ASE Atoms objects.
- **Outputs**: band gap in units of eV and formation energy per atom.
"""

NVIDIA_CARD = """---
license: apache-2.0
---
# Earth Checkpoints: OceanCast

## Description:
OceanCast predicts the evolution of the upper-ocean state for global ocean forecasting.

## Model Architecture:
**Architecture Type:** Spherical neural operator <br>
**Number of model parameters:** 120,500,000 <br>

## Input:
**Input Type:** Tensor (4 ocean variables) <br>
**Input Format:** PyTorch Tensor <br>
**Input Parameters:** Four Dimensional (4D) (batch, variable, latitude,
longitude) <br>
**Other Properties Related to Input:**
- Input latitude/longitude grid: 1 degree 180 x 360
- Input variables: `sst`, `sss`, `ssh`, `mld`

## Output:
**Output Type:** Tensor (4 ocean variables) <br>
**Output Format:** PyTorch Tensor <br>
**Other Properties Related to Output:**
- Output variables: same as input.

## Training:
The model is trained to produce 5-day forecasts and supports rollouts up to 365 days.
"""

FREEFORM_CARD = """---
license: cc-by-nc-4.0
datasets:
- example/ns-vortex
---

The base model of the Vortex family with approx. 85M parameters.

It is trained on a computational grid of size 64 x 64 and expects vorticity and velocity as inputs.

This is a research model.
"""

FREEFORM_CONFIG = {
    "architectures": ["VortexForPDE"],
    "num_channels": 3,
    "image_size": 64,
    "torch_dtype": "float32",
}

CONFIG_ONLY = {
    "architectures": ["GlitchClassifier"],
    "sampling_rate": 4096,
    "window_size": 8192,
    "id2label": {"0": "blip", "1": "koi fish", "2": "scattered light"},
    "n_params": 812345,
}

MOLECULE_TOOL = {
    "name": "molprop-qm",
    "server": "nomad",
    "description": "Predicts a molecular property for one SMILES string.",
    "input_schema": {
        "type": "object",
        "properties": {
            "smi": {
                "type": "string",
                "description": "SMILES string for one molecule",
            }
        },
        "required": ["smi"],
    },
}

MOLECULE_TOOL_CARD = """---
license: apache-2.0
tags:
- chemistry
- molecular-property-prediction
---

# MolProp-QM

MolProp-QM predicts quantum-chemical properties of small molecules from SMILES strings.

## Outputs

- **Outputs**: HOMO-LUMO gap regression in units of hartree.
"""

ROLLOUT_TOOL = {
    "name": "flowroll-2d",
    "server": "nomad",
    "description": "Rolls out a 2D flow state. Model dt = 0.05 s.",
    "input_schema": {
        "type": "object",
        "properties": {
            "duration": {
                "type": "integer",
                "exclusiveMinimum": 0,
                "description": "Number of time steps to rollout",
            },
            "initial_state": {
                "$ref": "#/$defs/WellFormat",
                "description": "Initial WellFormat state of the simulation",
            },
        },
        "required": ["duration", "initial_state"],
        "$defs": {
            "WellFormat": {
                "title": "WellFormat",
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"},
                    "n_spatial_dims": {"type": "integer"},
                    "t0_fields": {"type": "object"},
                },
            }
        },
    },
}

NATIVE_CARD_YAML = """
name: ToyMLIP
summary: Machine-learning interatomic potential for small organic crystals.
domains: [materials_atomistic]
tasks: [interatomic_potential]
license: apache-2.0
parameters: 3.1M
inputs:
  - name: structure
    kind: structure
    representation: ASE Atoms
outputs:
  - name: energy
    units: eV
  - name: forces
    units: eV/Å
    shape: (n_atoms, 3)
interfaces:
  - kind: python
    entry_point: toymlip:load
"""

TEMPLATE_RESIDUE_CARD = """---
license: {spdx_license_id}
tags:
- project:genesis
- science:lightsource
---

# ${MODEL_NAME}

## Inputs and outputs

(text, images, time series, etc.)

Examples of Inputs and Outputs:

1. Input: The model was trained with 224 x 224 input images and 512 token input/output text sequences.
2. Output: The output is a batch of feature vectors. For each input image, the feature vector has size num_features = 1280.

## Model License

If using a non-standard license (e.g. BSD, Apache2, MIT, etc...), please include it or a link to it here.

## Papers and Scientific Outputs

```bibtex
@article{Example2026,
  title = {{A Real Paper: Bibtex Braces Are Not Template Text}},
  year = {2026},
}
```
"""

FAMILY_TABLE_CARD = """---
license: apache-2.0
---
# Terra-EO

Terra-EO is a family of Earth observation foundation models.

| Model | Details |
|---|---|
| Terra-EO-tiny | Pretrained 5M parameter model |
| Terra-EO-300M | Pretrained 300M parameter model |
| Terra-EO-600M | Pretrained 600M parameter model |
"""

SHARED_METRICS_CARD = """---
datasets: example/rayleigh_benard
---

# Benchmarking surrogate models

The reported results are a simple baseline for PDE surrogate modeling.

| Dataset | Epochs | VRMSE |
|---|---|---|
| shear_flow | 24 | 0.4450 |
| rayleigh_benard | 32 | 0.6049 |
| active_matter | 239 | 0.3157 |
| planetswe | 49 | 0.0855 |
"""

LICENSE_CONFLICT_CARD = """---
license: apache-2.0
tags:
- chemistry
---

# ChemNet

ChemNet predicts reaction yields from SMILES strings of reactants and products, for chemistry research.

## Use and Restrictions

Model weights are provided for research purposes only.
- Research use only
- No commercial use without licensing agreement
"""

# A served property predictor in the style of Nomad's MIST cards: outputs in
# an "Outputs" bullet and a code comment, a fine-tuning paragraph with the
# data size and range, and a base model.
SERVED_PROPERTY_CARD = """---
license: apache-2.0
tags:
- chemistry
---

# QuantaProp: molecular property transformers

QuantaProp is a family of molecular foundation models for property prediction.

## Model Details

This fine-tuned variant pairs the pretrained encoder with a task network.
The final hidden states are pooled to produce a single embedding vector.

- **Finetuned from model:** `quanta-base-20M`

## Getting Started

```python
model = AutoModel.from_pretrained("path/to/model", trust_remote_code=True)
# Returns predictions for 3 quantum properties (homo, lumo, gap)
results = model.predict(["CCO"])
```

### Use and Restrictions

- Research use only
- No commercial use without licensing agreement

## Training Details

### Training Data

**Finetuning** The variant was fine-tuned on a dataset of 1,200 molecules (up to 9 heavy atoms) with DFT labels.

#### Inputs

- **Inputs**: SMILES strings for molecules.
- **Outputs**: Regression predictions for 3 quantum properties
"""
SERVED_PROPERTY_SCHEMA = {
    "type": "object",
    "properties": {
        "smi": {
            "type": "string",
            "description": "SMILES string for one molecule",
        }
    },
    "required": ["smi"],
}
