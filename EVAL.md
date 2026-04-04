# OpenVLA-OFT Augmented Prompt Evaluation

Evaluate the OpenVLA-OFT checkpoint (finetuned on all 4 LIBERO suites) against 8 augmentation types + original prompts.

## Setup

```bash
# 1. Create conda environment
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# 2. Install PyTorch (use command for your CUDA version from https://pytorch.org)
pip3 install torch torchvision torchaudio

# 3. Install openvla-oft
cd /path/to/openvla-oft
pip install -e .

# 4. Install Flash Attention 2 (needed for model loading)
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

# 5. Install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO

# 6. Install LIBERO eval dependencies
pip install -r experiments/robot/libero/libero_requirements.txt

# 7. Install additional dependencies
pip install tensorflow==2.15.0 json-numpy "numpy<2"
```

## What gets evaluated

- **Model**: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` (all 4 suites combined)
- **Task suites**: libero_goal, libero_spatial, libero_object, libero_10 (10 tasks each)
- **Augmentation types**: original, verb_synonym, object_synonym, phrasing, synonym_combined, synonym_mixed, logical, goal_state, hierarchical
- **Trials**: 50 per task (500 per suite per augmentation type)
- **Prompt sampling**: Each episode randomly samples one holdout prompt from a pool of 10

Total: 4 suites x 9 aug types = 36 runs x 500 episodes = 18,000 episodes.

## Running evaluations

### Option A: Shell script

```bash
conda activate openvla-oft
bash scripts/run_all_evals_augmented.sh
```

The script skips completed runs (uses `.done` marker files), so it's safe to restart.

### Option B: Manual commands

Use the full path to the conda Python to avoid PATH issues with other virtualenvs:

```bash
cd /path/to/openvla-oft
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/openvla-oft

PYTHON=/path/to/miniconda3/envs/openvla-oft/bin/python
CKPT="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
YAML="data/holdout_yamls"
LOG="experiments/evals/openvla_oft_augmented/logs"
ARGS="--pretrained_checkpoint $CKPT --num_trials_per_task 50 --use_l1_regression True --use_proprio True --num_images_in_input 2 --local_log_dir $LOG"

mkdir -p $LOG

# Original prompts (no holdout YAML)
$PYTHON experiments/robot/libero/run_libero_eval.py $ARGS --task_suite_name libero_goal --run_id_note original

# Augmented prompts (with holdout YAML)
$PYTHON experiments/robot/libero/run_libero_eval.py $ARGS --task_suite_name libero_goal --run_id_note verb_synonym --holdout_prompts_yaml $YAML/verb_synonym.yaml
```

Chain with `&&` for all combinations. Available augmentation YAMLs:
`verb_synonym.yaml`, `object_synonym.yaml`, `phrasing.yaml`, `synonym_combined.yaml`, `synonym_mixed.yaml`, `logical.yaml`, `goal_state.yaml`, `hierarchical.yaml`

Available suites: `libero_goal`, `libero_spatial`, `libero_object`, `libero_10`

## Output

Logs are written to `experiments/evals/openvla_oft_augmented/logs/`. Each run produces a text file with per-task success rates.

## Data files

- `data/holdout_yamls/*.yaml` — 8 YAML files, one per augmentation type. Each maps 40 original task descriptions to 10 holdout (test-split) prompts. These are the test split from `augmented_prompts.json`, used for evaluation only (not seen during training).
