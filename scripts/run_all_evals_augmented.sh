#!/bin/bash
# Run all augmentation type x task suite evaluations for OpenVLA-OFT finetuned checkpoint.
# Usage: bash scripts/run_all_evals_augmented.sh
#
# Prerequisites:
#   conda activate openvla-oft
#   cd /home/fhliang/projects/openvla-oft
#   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# All outputs go here — logs and rollouts
EVAL_DIR="$REPO_ROOT/experiments/evals/openvla_oft_augmented"
LOG_DIR="$EVAL_DIR/logs"
ROLLOUT_DIR="$EVAL_DIR/rollouts"
mkdir -p "$LOG_DIR" "$ROLLOUT_DIR"

# Must run from here so sys.path.append("../..") resolves experiments package
cd "$REPO_ROOT/experiments/robot/libero"
# Symlink rollouts dir so save_rollout_video()'s hardcoded ./rollouts/ path lands in EVAL_DIR
ln -sfn "$ROLLOUT_DIR" rollouts

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CHECKPOINT="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
YAML_DIR="$REPO_ROOT/data/holdout_yamls"
TRIALS=50
SUITES="libero_goal libero_spatial libero_object libero_10"
AUG_TYPES="original verb_synonym object_synonym phrasing synonym_combined synonym_mixed logical goal_state hierarchical"

mkdir -p "$LOG_DIR"

for SUITE in $SUITES; do
    for AUG in $AUG_TYPES; do
        NAME="openvla_oft_${AUG}_${SUITE}"
        DONE_MARKER="${LOG_DIR}/${NAME}.done"

        if [ -f "$DONE_MARKER" ]; then
            echo "==> Skipping $NAME (already done)"
            continue
        fi

        echo ""
        echo "=========================================="
        echo "Running: $AUG / $SUITE ($TRIALS trials/task)"
        echo "=========================================="

        if [ "$AUG" = "original" ]; then
            python run_libero_eval.py \
                --pretrained_checkpoint "$CHECKPOINT" \
                --task_suite_name "$SUITE" \
                --num_trials_per_task "$TRIALS" \
                --use_l1_regression True \
                --use_proprio True \
                --num_images_in_input 2 \
                --run_id_note "$AUG" \
                --local_log_dir "$LOG_DIR"
        else
            python run_libero_eval.py \
                --pretrained_checkpoint "$CHECKPOINT" \
                --task_suite_name "$SUITE" \
                --num_trials_per_task "$TRIALS" \
                --use_l1_regression True \
                --use_proprio True \
                --num_images_in_input 2 \
                --run_id_note "$AUG" \
                --holdout_prompts_yaml "${YAML_DIR}/${AUG}.yaml" \
                --local_log_dir "$LOG_DIR"
        fi

        touch "$DONE_MARKER"
    done
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Logs in: $LOG_DIR/"
echo "=========================================="
