from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Union
import transformers
from transformers.utils import PaddingStrategy

@dataclass
class ProjectArguments:
    random_state: int = field(
        default=42,
        metadata={"help": "random state"}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )
    wandb_project: Optional[str] = field(
        default="Test"
    )
    wandb_name: Optional[str] = field(
        default="opt-1.3b"
    )
    save_dir: Optional[str] = field(
        default="/home/lune/nas2/lightning/lightning-template/checkpoint"
    )
    save_prediction_file: Optional[str] = field(
        default="/home/lune/nas2/lightning/lightning-template/result"
    )
    strategy: Optional[str] = field(
        default='deepspeed_stage_2'
    )
    
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="facebook/opt-1.3b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    model_4bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Use 4bit model."}
    )

    
@dataclass
class DataArguments:
    rawdata_save_dir: str = field(
        default='/home/lune/nas2/data/prompts',
        metadata = {
            "help": "default rawdata save dir"
        }
    )
    dataset_name: str = field(
        default='alespalla/chatbot_instruction_prompts',
        metadata = {
            "help": "default dataset name for loading"
        }
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    source_max_len: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    truncation_side: str = field(
        default='right',
        metadata = {
            "help": "truncation side"
        }
    )
    padding: Union[bool, str, PaddingStrategy]= field(
        default=True,
        metadata={"help": "Padding Strategy."},
    )
    pad_to_multiple_of: Optional[int] = field(
        default=None,
        metadata = {
            "help": "This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=7.5"
        }        
    )

@dataclass
class SchedulingArguments:
    lr: Optional[float] = field(
        default=0.0001
    )
    max_lr: Optional[float] = field(
        default=0.0002
    )
    use_lrfinder: bool = field(
        default = False
    )
    step_size_up: Optional[int] = field(
        default=1000
    )
    scheduler_type: Optional[str] = field(
        default='ReduceLROnPlateau'
    )
    scheduler_mode: Optional[str] = field(
        default='min'
    )
    factor: Optional[float] = field(
        default=0.1
    )
    patience: Optional[int] = field(
        default=2
    )
    
    
@dataclass
class TrainingArguments:
    devices: Optional[List[int]] = field(
        default_factory=lambda: [0]
    )
    do_train: Optional[bool] = field(
        default=True
    )
    do_test: Optional[bool] = field(
        default=True
    )
    max_epochs: Optional[int] = field(
        default=50
    )
    per_device_train_batch_size: Optional[int] = field(
        default=None
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=None
    )
    predict_with_generate: Optional[bool] = field(
        default=False
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )    
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )    


@dataclass
class LoRaArguments:
    max_memory_MB: int = field(
        default=20000,
        metadata={"help": "Free memory per gpu."}
    )    
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    fp16: Optional[bool] = field(
        default=False
    )    
    bf16: Optional[bool] = field(
        default=False
    )   
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})

    # report_to: str = field(
    #     default='none',
    #     metadata={"help": "To use wandb or something else for reporting."}
    # )
    # output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    # max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    # weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    # remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    # max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    # gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})

    # warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    # save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    # save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})


@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=700,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True) 

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=0.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0) 
    repetition_penalty: Optional[float] = field(default=1.0) 
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0) 