#!/bin/bash

set -e

REPO_URL="https://github.com/v0dnyy/air-vision-module.git"
SCRIPT_TO_RUN="inference.py"
SCRIPT_ARGS=(
    --path_to_model_w "yolo_n_v11_dropout_best.pt"
    --from_cam
    --save_video
    --save_logs
)

echo "--- Начало установки и запуска ---"

if [ -d "air-vision-module" ]; then
    echo "Директория 'air-vision-module' уже существует, обновляю содержимое..."
    cd "air-vision-module"
    git pull origin main
else
    echo "Клонируем репозиторий из $REPO_URL"
    git clone "$REPO_URL"
    cd "air-vision-module"
fi

echo "Создание нового Conda окружения..."
# Удаляем существующее окружение, если оно есть
conda env remove -n air-vision-module 2>/dev/null || true

# Создаем новое Conda окружение с Python 3.10
conda create -n air-vision-module python=3.10 -y

# Активируем Conda окружение
source $(conda info --base)/etc/profile.d/conda.sh
conda activate air-vision-module

echo "Устанавливаем зависимости из requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Файл requirements.txt не найден!"
    exit 1
fi

echo "Устанавливаем специфические зависимости..."
pip install ultralytics[export]

pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev

pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
pip install onnx==1.15.0
pip install numpy==1.23.5

echo "Запускаем скрипт $SCRIPT_TO_RUN..."

python "$SCRIPT_TO_RUN" "${SCRIPT_ARGS[@]}"

echo "=== Скрипт завершил работу ==="
conda deactivate

