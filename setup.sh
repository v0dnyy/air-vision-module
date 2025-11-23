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

echo "Создание нового виртуального окружения air-vision-module-venv..."
if [ -d "air-vision-module-venv" ]; then
    echo "Виртуальное окружение 'air-vision-module-venv' уже существует, удаляю..."
    rm -rf air-vision-module-venv
fi

python3 -m venv air-vision-module-venv
echo "Активация виртуального окружения air-vision-module-venv..."
source air-vision-module-venv/bin/activate

echo "Устанавливаем зависимости из requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Файл requirements.txt не найден!"
    deactivate
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
deactivate

