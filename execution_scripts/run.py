import sys
from pathlib import Path

# Корень проекта — чтобы импортировать constants, models и core из любого места
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from constants import EPOCHS
from models.fc import (
    LinearClassifier,
    OneHiddenLayer,
    WideHidden,
    MultiLayer,
    TinyBottleneck,
    DeepNarrow,
)
from models.cnn import (
    ConvOneLayer,
    ConvTwoLayers,
    ConvDeep,
    ConvGlobalPool,
)
from core.train import train
from core.inference import evaluate
from core.sizes import (
    save_model,
    model_size_bytes,
    dataset_size_bytes,
    compression_ratio,
)

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
RESULTS_PATH = PROJECT_ROOT / "results.py"


def run_one(name, model, train_loader, test_loader, device, train_dir):
    """Один эксперимент: обучение, инференс, сохранение, коэффициент сжатия."""
    print(f"\n=== Модель: {name} ===")
    model = train(model, train_loader, device, EPOCHS, lr=LEARNING_RATE)

    test_accuracy = evaluate(model, test_loader, device)
    print(f"Точность на тестовой выборке: {test_accuracy:.4f}")

    model_path = save_model(model, PROJECT_ROOT / "saved_models" / f"{name}.pth")
    model_bytes = model_size_bytes(model_path)
    dataset_bytes = dataset_size_bytes(train_dir)
    ratio = compression_ratio(dataset_bytes, model_bytes)

    print(f"Размер модели (.pth): {model_bytes} байт")
    print(f"Коэффициент сжатия:   {ratio:.2f}")
    return {"name": name, "accuracy": test_accuracy, "bytes": model_bytes, "ratio": ratio}


def run_fc_models(train_loader, test_loader, device, train_dir):
    """Эксперимент для полносвязных моделей."""
    experiments = [
        ("linear_classifier", LinearClassifier()),
        ("one_hidden_layer", OneHiddenLayer()),
        ("wide_hidden", WideHidden()),
        ("multi_layer", MultiLayer()),
        ("tiny_bottleneck", TinyBottleneck()),
        ("deep_narrow", DeepNarrow()),
    ]
    results = []
    for name, model in experiments:
        results.append(run_one(name, model, train_loader, test_loader, device, train_dir))
    return results


def run_conv_models(train_loader, test_loader, device, train_dir):
    """Эксперимент для свёрточных моделей."""
    experiments = [
        ("conv_one_layer", ConvOneLayer()),
        ("conv_two_layers", ConvTwoLayers()),
        ("conv_deep", ConvDeep()),
        ("conv_global_pool", ConvGlobalPool()),
    ]
    results = []
    for name, model in experiments:
        results.append(run_one(name, model, train_loader, test_loader, device, train_dir))
    return results


def save_results(results, path):
    """Сохраняет результаты на тестовых выборках в results.py.

    Новые модели дописываются после уже сохранённых. Если модель с таким
    именем уже есть — её запись заменяется последней попыткой.
    """
    # Уже сохранённые результаты прошлых запусков
    try:
        from results import RESULTS as existing
    except Exception:
        existing = []

    # Новые результаты в формате хранения (ключ size_bytes вместо bytes)
    new_records = [
        {
            "name": r["name"],
            "accuracy": r["accuracy"],
            "size_bytes": r["bytes"],
            "ratio": r["ratio"],
        }
        for r in results
    ]

    # Обновляем по имени: существующую запись заменяем, новую дописываем в конец
    all_records = list(existing)
    index_by_name = {rec["name"]: i for i, rec in enumerate(all_records)}
    for rec in new_records:
        if rec["name"] in index_by_name:
            all_records[index_by_name[rec["name"]]] = rec
        else:
            index_by_name[rec["name"]] = len(all_records)
            all_records.append(rec)

    lines = [
        '"""Результаты экспериментов на тестовых выборках.\n',
        "\n",
        "Файл дополняется при запуске execution_scripts/run.py:\n",
        "новые результаты добавляются после уже сохранённых.\n",
        "Каждая запись: имя модели, точность на test, размер .pth в байтах, коэффициент сжатия.\n",
        '"""\n',
        "\n",
        "RESULTS = [\n",
    ]
    for rec in all_records:
        lines.append(
            f'    {{"name": "{rec["name"]}", '
            f'"accuracy": {rec["accuracy"]:.4f}, '
            f'"size_bytes": {rec["size_bytes"]}, '
            f'"ratio": {rec["ratio"]:.2f}}},\n'
        )
    lines.append("]\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    # Загрузка выборок из картинок (ImageFolder).
    # ImageFolder отдаёт RGB, поэтому возвращаем картинку в 1 канал.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    data_dir = PROJECT_ROOT / "datasets" / "data" / "mnist_images"
    train_dir = data_dir / "train"
    train_set = datasets.ImageFolder(root=str(train_dir), transform=transform)
    test_set = datasets.ImageFolder(root=str(data_dir / "test"), transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Устройство: видеокарта Apple (MPS), иначе CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Устройство: {device}")

    # закоментировать строки при за ненадобностью
    results = []
    results += run_fc_models(train_loader, test_loader, device, train_dir)
    results += run_conv_models(train_loader, test_loader, device, train_dir)

    # Сводка по эксперименту
    print("\n=== Сводка ===")
    for r in results:
        print(
            f"{r['name']:>18}  точность={r['accuracy']:.4f}  "
            f"размер={r['bytes']} байт  коэффициент={r['ratio']:.2f}"
        )

    # Сохранение результатов в results.py
    save_results(results, RESULTS_PATH)
    print(f"\nРезультаты сохранены в {RESULTS_PATH}")


if __name__ == "__main__":
    main()
