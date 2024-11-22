from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .util import MODEL_CHECKPOINT, SOURCE_DOMAIN, TARGET_DOMAIN


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "largeQA" / "data"


@dataclass
class ExperimentConfig:
    source_domain: str = SOURCE_DOMAIN
    target_domain: str = TARGET_DOMAIN
    model_checkpoint: str = MODEL_CHECKPOINT
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = REPO_ROOT / "checkpoints"
    batch_size: int = 4
    num_train_epochs: int = 1
    learning_rate: float = 5.6e-5
    weight_decay: float = 0.01
    train_subset: Optional[int] = None
    prompt_length: int = 100
    max_input_length: int = 512


def resolve_data_dir(data_root: Path | str, domain: str, split: str) -> Path:
    return Path(data_root).expanduser().resolve() / f"{domain}_{split}"


def model_slug(model_checkpoint: str) -> str:
    return model_checkpoint.split("/")[-1]


def maybe_set_cuda_visible_devices(devices: Optional[str]) -> None:
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices


def pipeline_device() -> int:
    return 0 if torch.cuda.is_available() else -1
