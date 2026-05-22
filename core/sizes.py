from pathlib import Path

import torch


def save_model(model, path):
    """Сохраняет веса модели в .pth файл. Возвращает путь к файлу."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def model_size_bytes(path):
    """Размер .pth файла модели в байтах."""
    return Path(path).stat().st_size


def dataset_size_bytes(directory):
    """Суммарный размер всех файлов в директории датасета в байтах."""
    total = 0
    for file in Path(directory).rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total


def compression_ratio(dataset_bytes, model_bytes):
    """Коэффициент сжатия: размер датасета / размер модели (в байтах)."""
    return dataset_bytes / model_bytes
