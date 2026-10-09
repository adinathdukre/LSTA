#!/bin/bash
cd "$(dirname "$0")/.."
rank=(8)
k=(8)
outer_lr=(1e-4)
eta=(2 4 8 16)
T=(1.0)
seeds=(42 43 44 45 46)
python -u tta.py PTB-XL --seeds "${seeds[@]}" --out_root runs/TTA \
    --eta_grid "${eta[@]}" --rank_grid "${rank[@]}" --k_grid "${k[@]}" \
    --outer_lr_grid "${outer_lr[@]}" --adapt_steps_grid 1 \
    --temperature_grid "${T[@]}" --gpu "${GPU:-4}" --tune_subjects 400 "$@"
