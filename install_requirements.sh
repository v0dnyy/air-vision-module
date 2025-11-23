#!/bin/bash

set -e

echo "--- Начало установки зависимостей ---"
echo "Создание нового виртуального окружения air-vision-module-venv..."
if [ -d "air-vision-module-venv" ]; then
    echo "Виртуальное окружение 'air-vision-module-venv' уже существует, удаляю..."
    rm -rf air-vision-module-venv
fi

python3 -m venv \air-vision-module-venv
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