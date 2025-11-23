#!/bin/bash

set -e
SCRIPT_TO_RUN="inference.py"
SCRIPT_ARGS=(
    --path_to_model_w "yolo_n_v11_dropout_best.pt"
    --from_cam
    --cam_num 0
    --show_video
    --save_video
    --save_logs
)

echo "Запускаем скрипт $SCRIPT_TO_RUN..."

python "$SCRIPT_TO_RUN" "${SCRIPT_ARGS[@]}"

echo "=== Скрипт завершил работу ==="