# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is OpenVLA-OFT (Open Vision-Language-Action with Optimized Fine-Tuning), a research project for fine-tuning vision-language-action models for robotics tasks. The codebase implements optimized fine-tuning techniques including LoRA, action head optimization, and parallel decoding to achieve faster training and better performance on robotic manipulation tasks.

**Key Research Focus**: This project implements **Quality Diversity (QD) adversarial prompt training** to improve robustness of vision-language-action models. The system trains models on both original task instructions and QD-generated adversarial prompts that are semantically different but should produce similar robot behaviors, making the models less sensitive to instruction phrasing variations.

## Architecture

### Core Components

**VLA Models (`prismatic/models/vlas/`)**
- `openvla.py`: Main OpenVLA model extending PrismaticVLM with action prediction capabilities
- Action tokenization and continuous action prediction via regression or diffusion heads

**Action Prediction (`prismatic/models/action_heads.py`)**
- `L1RegressionActionHead`: Direct regression for continuous actions
- `DiffusionActionHead`: Diffusion-based action generation using DDIM
- Supports multi-step action chunks for temporal consistency

**Vision-Language Backbone (`prismatic/models/`)**
- Modular vision encoders: SigLIP, CLIP, DINOv2, DinoSigLIP variants
- LLM backbones: Llama-2, Mistral, Phi support
- Projectors for mapping between vision/language/action spaces

**Dataset Handling (`prismatic/vla/datasets/`)**
- RLDS (Robot Learning Dataset) format support
- OXE (Open X-Embodiment) dataset mixtures
- Platform-specific constants: LIBERO (sim), ALOHA (real robot), Bridge (real robot)
- **QD Adversarial Prompt System**: `datasets2.py` implements adversarial prompt injection during training using YAML-configured alternative task descriptions

**Training Infrastructure (`prismatic/training/`)**
- FSDP and DDP distributed training strategies
- Mixed precision training with gradient checkpointing
- Custom learning rate schedulers and optimization

### Key Constants & Configuration

**Robot Platform Detection (`prismatic/vla/constants.py`)**
- Automatically detects robot platform from command line arguments
- LIBERO: 8-step chunks, 7D actions, 8D proprioception
- ALOHA: 25-step chunks, 14D actions, 14D proprioception  
- BRIDGE: 5-step chunks, 7D actions, 7D proprioception

**Training Configurations (`prismatic/conf/vla.py`)**
- Predefined experiment configs for different model sizes and datasets
- Supports freezing vision/LLM backbones for parameter-efficient fine-tuning
- OXE Magic Soup datasets for large-scale pretraining

## Common Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# Install PyTorch (check pytorch.org for your specific CUDA version)
pip3 install torch torchvision torchaudio

# Install this package and dependencies
pip install -e .

# Install Flash Attention 2 for training
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

### Fine-tuning
```bash
# Basic LoRA fine-tuning (uses datasets2.py with configurable adversarial prompts)
python vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --dataset_name "aloha_scoop_x_into_bowl" \
    --run_root_dir "./runs" \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --max_steps 200_000 \
    --adv_prompts_yaml "my_custom_prompts.yaml" \
    --adv_replace_prob 0.3 \
    --adv_log_examples_every 100

# Advanced QD adversarial fine-tuning (configurable via YAML)
python vla-scripts/adv_finetune.py \
    --use_diffusion true \
    --use_film true \
    --num_images_in_input 2 \
    --use_proprio true \
    --adv_prompts_path "new_task_descriptions.yaml" \
    --adv_replace_prob 0.5 \
    --adv_log_examples_every 50
```

### Evaluation

**LIBERO Simulation**
```bash
# Setup LIBERO environment
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt

# Run evaluation
python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint "moojink/openvla-7b-oft-finetuned-libero-spatial" \
    --task_suite_name "libero_spatial" \
    --num_trials_per_task 50
```

**ALOHA Real Robot**
```bash
# Install ALOHA dependencies
pip install -r experiments/robot/aloha/requirements_aloha.txt

# Deploy on robot
python vla-scripts/deploy.py \
    --checkpoint_path "./runs/your_checkpoint" \
    --robot_type "aloha"
```

### Code Quality
```bash
# Format code with Black
black . --line-length 121

# Lint with Ruff  
ruff check . --fix

# Run pre-commit hooks
pre-commit run --all-files
```

### Model Weights Management
```bash
# Merge LoRA weights with base model
python vla-scripts/merge_lora_weights_and_save.py \
    --checkpoint_path "./runs/your_lora_checkpoint" \
    --output_path "./merged_model"

# Convert to HuggingFace format
python vla-scripts/extern/convert_openvla_weights_to_hf.py \
    --model_path "./merged_model" \
    --output_path "./hf_model"
```

## Development Guidelines

### QD Adversarial Prompt Training
- **Purpose**: Improve model robustness by training on diverse phrasings of the same task
- **Configuration**: Use `adv_replace_prob` (0.0-1.0) to control frequency of adversarial prompt substitution
- **Logging**: Set `adv_log_examples_every` to monitor prompt replacements during training
- **YAML Format**: Define alternative prompts in YAML files with original task as key, alternatives as list
- **Two Implementations**: 
  - `finetune.py` + `datasets2.py`: Fast adversarial system with configurable YAML paths
  - `adv_finetune.py` + `datasets.py`: Modular system with advanced prompt sampling features

### Action Head Selection
- Use `L1RegressionActionHead` for direct, fast action prediction
- Use `DiffusionActionHead` for multi-modal action distributions
- Configure via `use_l1_regression` and `use_diffusion` flags

### Multi-Step Action Prediction
- Action chunks provide temporal consistency and planning capability
- Chunk sizes are platform-specific (see constants.py)
- Use `num_open_loop_steps` to control execution horizon

### Memory Optimization
- Enable gradient checkpointing for large models: `enable_gradient_checkpointing=True`
- Use mixed precision training: `enable_mixed_precision_training=True`  
- Reduce batch size if encountering OOM errors
- Consider freezing vision backbone for parameter-efficient fine-tuning

### Dataset Integration
- RLDS format required for all datasets
- Use `RLDSDataset` class for data loading
- Configure normalization via `ACTION_PROPRIO_NORMALIZATION_TYPE`
- Set appropriate `shuffle_buffer_size` based on dataset size

### Distributed Training
- Use FSDP for multi-GPU training: `train_strategy="fsdp-full-shard"`
- Configure `expected_world_size`, `global_batch_size`, `per_device_batch_size`
- Gradient accumulation steps computed automatically

### Model Loading and Inference
- Models available on HuggingFace Hub (moojink/openvla-7b-oft-finetuned-*)
- Use `experiments.robot.openvla_utils` for inference utilities
- Support for 4-bit and 8-bit quantization via `load_in_4bit`/`load_in_8bit`

## Project Structure

```
prismatic/
├── models/           # Model architectures and components
│   ├── vlas/        # VLA-specific models (OpenVLA)
│   ├── action_heads.py  # Action prediction heads
│   ├── projectors.py    # Cross-modal projectors
│   └── backbones/   # Vision and LLM backbones
├── vla/             # VLA-specific code
│   ├── datasets/    # Dataset handling and RLDS processing
│   ├── constants.py # Platform-specific constants
│   └── action_tokenizer.py  # Action discretization
├── training/        # Training infrastructure
├── conf/           # Configuration classes
└── extern/         # HuggingFace integration

vla-scripts/        # Training and evaluation scripts
├── finetune.py     # Basic LoRA fine-tuning with hardcoded QD adversarial prompts
├── adv_finetune.py # Advanced fine-tuning with configurable QD adversarial prompts + diffusion/FiLM  
├── deploy.py       # Robot deployment script
└── test.py         # Dataset inspection utilities

experiments/robot/  # Robot-specific evaluation code
├── libero/         # LIBERO simulation evaluation
├── aloha/          # ALOHA real robot evaluation  
└── openvla_utils.py # OpenVLA inference utilities
```