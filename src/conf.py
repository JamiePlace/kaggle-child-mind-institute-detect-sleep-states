from dataclasses import dataclass
from typing import Any


@dataclass
class DirConfig:
    data_dir: str
    processed_dir: str
    output_dir: str
    model_dir: str
    sub_dir: str


@dataclass
class SplitConfig:
    name: str
    train_series_ids: list[str]
    valid_series_ids: list[str]


@dataclass
class FeatureExtractorConfig:
    name: str
    params: dict[str, Any]


@dataclass
class DecoderConfig:
    name: str
    params: dict[str, Any]


@dataclass
class ModelConfig:
    name: str
    params: dict[str, Any]


@dataclass
class TrainerConfig:
    epochs: int
    accelerator: str
    use_amp: bool
    debug: bool
    gradient_clip_val: float
    accumulate_grad_batches: int
    monitor: str
    monitor_mode: str
    check_val_every_n_epoch: int
    early_stopping_patience: int


@dataclass
class DatasetConfig:
    batch_size: int
    num_workers: int
    offset: int
    sigma: int
    bg_sampling_rate: float
    positive_to_negative_ratio: float
    window_size: int
    batch_size: int
    on_training: bool
    series_ids: list[str]


@dataclass
class AugmentationConfig:
    mixup_prob: float
    mixup_alpha: float
    cutmix_prob: float
    cutmix_alpha: float


@dataclass
class PostProcessConfig:
    score_th: float
    distance: int


@dataclass
class OptimizerConfig:
    lr: float


@dataclass
class SchedulerConfig:
    num_warmup_steps: int
    num_cycles: int
    power: int


@dataclass
class WeightConfig:
    exp_name: str
    run_name: str


@dataclass
class TrainConfig:
    exp_name: str
    seed: int
    duration: int
    subsample: bool
    subsample_rate: float
    downsample_rate: int
    upsample_rate: int
    labels: list[str]
    features: list[str]
    split: SplitConfig
    dir: DirConfig
    model: ModelConfig
    feature_extractor: FeatureExtractorConfig
    decoder: DecoderConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    dataset: DatasetConfig
    aug: AugmentationConfig
    pp: PostProcessConfig
    refresh_processed_data: bool
    dense_weight: float


@dataclass
class InferenceConfig:
    exp_name: str
    phase: str
    seed: int
    batch_size: int
    num_workers: int
    duration: int
    downsample_rate: int
    upsample_rate: int
    use_amp: bool
    labels: list[str]
    features: list[str]
    dir: DirConfig
    model: ModelConfig
    feature_extractor: FeatureExtractorConfig
    decoder: DecoderConfig
    weight: WeightConfig
    aug: AugmentationConfig
    pp: PostProcessConfig
    dataset: DatasetConfig
    series_ids: list[str]
    event_threshold: int
    duration_threshold: int
    prediction_threshold: float
    all_training: bool


@dataclass
class PrepareDataConfig:
    dir: DirConfig
    phase: str
    dataset: DatasetConfig
    features: list[str]
    split: SplitConfig
