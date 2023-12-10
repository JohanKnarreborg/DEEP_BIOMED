from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from typing import Any, List, Optional

def register_config():
    ConfigStore.instance().store(name="config_schema", node=ConfigSchema)

@dataclass
class DataConfig:
    data_path: str = MISSING
    input_img_size: int = MISSING
    crop_volume_size: int = MISSING
    prob_foreground_center: float = MISSING # Probability that center of crop is a labeled foreground voxel (ensures the crops often contain a label)


@dataclass
class TrainingConfig:
    device: str = MISSING
    train_batch_size: int = MISSING
    val_batch_size: int = MISSING
    batches_per_epoch: int = MISSING
    num_epochs: int = MISSING
    loss_fn_name: str = MISSING


@dataclass
class OptimizerConfig:
    learning_rate: float = MISSING


@dataclass
class ModelConfig:
    model_type: str = MISSING
    pretrained_model: Optional[str] = None
    freeze_encoder: Optional[bool] = None


@dataclass
class ConfigSchema:
    run_name: str = MISSING
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    model: ModelConfig = ModelConfig()
    hydra: Any = field(default_factory=lambda: {"job": {"chdir": False}, "run": {"dir": "."}})
    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"override hydra/hydra_logging": "none"},
            {"override hydra/job_logging": "none"},
        ]
    )