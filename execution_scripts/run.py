import sys
import importlib
from pathlib import Path

# Корень проекта — чтобы импортировать constants, models и core из любого места
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from constants import EPOCHS, DISTILL_T, DISTILL_ALPHA
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
from core.quantization import quantize_dynamic
from core.distillation import train_distilled

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
RESULTS_BASELINE_PATH = PROJECT_ROOT / "results.py"
RESULTS_QUANTIZED_PATH = PROJECT_ROOT / "results_quantized" / "__init__.py"
RESULTS_DISTILLED_PATH = PROJECT_ROOT / "results_distilled" / "__init__.py"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
QUANTIZED_DIR = SAVED_MODELS_DIR / "quantized"
DISTILLED_DIR = SAVED_MODELS_DIR / "distilled"


def run_one(name, model, train_loader, test_loader, device, train_dir):
    """Один эксперимент: обучение, инференс, сохранение, коэффициент сжатия."""
    print(f"\n=== Модель: {name} ===")
    model = train(model, train_loader, device, EPOCHS, lr=LEARNING_RATE)

    test_accuracy = evaluate(model, test_loader, device)
    print(f"Точность на тестовой выборке: {test_accuracy:.4f}")

    model_path = save_model(model, SAVED_MODELS_DIR / f"{name}.pth")
    model_bytes = model_size_bytes(model_path)
    dataset_bytes = dataset_size_bytes(train_dir)
    ratio = compression_ratio(dataset_bytes, model_bytes)

    print(f"Размер модели (.pth): {model_bytes} байт")
    print(f"Коэффициент сжатия:   {ratio:.2f}")
    return {
        "name": name,
        "kind": "baseline",
        "accuracy": test_accuracy,
        "bytes": model_bytes,
        "ratio": ratio,
    }


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
    return [
        run_one(name, m, train_loader, test_loader, device, train_dir)
        for name, m in experiments
    ]


def run_conv_models(train_loader, test_loader, device, train_dir):
    """Эксперимент для свёрточных моделей."""
    experiments = [
        ("conv_one_layer", ConvOneLayer()),
        ("conv_two_layers", ConvTwoLayers()),
        ("conv_deep", ConvDeep()),
        ("conv_global_pool", ConvGlobalPool()),
    ]
    return [
        run_one(name, m, train_loader, test_loader, device, train_dir)
        for name, m in experiments
    ]


def run_quantization_experiment(test_loader, train_dir):
    """Dynamic int8 квантизация для всех моделей.

    Берёт обученные .pth из saved_models, применяет
    torch.quantization.quantize_dynamic к Linear-слоям, оценивает на CPU,
    сохраняет в saved_models/quantized/.

    Для свёрточных моделей dynamic quant трогает только финальный
    Linear-слой (свёрточные слои остаются float). ConvGlobalPool, у которой
    Linear нет вовсе, по сути не квантизуется.
    """
    models_to_quantize = {
        "linear_classifier": LinearClassifier,
        "one_hidden_layer": OneHiddenLayer,
        "wide_hidden": WideHidden,
        "multi_layer": MultiLayer,
        "tiny_bottleneck": TinyBottleneck,
        "deep_narrow": DeepNarrow,
        "conv_one_layer": ConvOneLayer,
        "conv_two_layers": ConvTwoLayers,
        "conv_deep": ConvDeep,
        "conv_global_pool": ConvGlobalPool,
    }
    cpu = torch.device("cpu")
    QUANTIZED_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for name, cls in models_to_quantize.items():
        source_path = SAVED_MODELS_DIR / f"{name}.pth"
        if not source_path.exists():
            print(f"\n=== Квантизация: {name} ===  ПРОПУСК (нет {source_path.name})")
            continue

        print(f"\n=== Квантизация: {name} ===")
        # Загружаем обученные веса в свежую модель на CPU
        model = cls()
        model.load_state_dict(torch.load(source_path, map_location="cpu"))

        # Dynamic int8 квантизация Linear-слоёв
        quantized = quantize_dynamic(model)

        # Оценка квантизованной модели — обязательно на CPU
        accuracy = evaluate(quantized, test_loader, cpu)
        print(f"Точность на тестовой выборке: {accuracy:.4f}")

        # Сохраняем целиком: структура модели изменилась (QuantizedLinear),
        # state_dict без структуры обратно не загрузить без возни.
        quant_path = QUANTIZED_DIR / f"{name}.pth"
        torch.save(quantized, quant_path)

        model_bytes = model_size_bytes(quant_path)
        dataset_bytes = dataset_size_bytes(train_dir)
        ratio = compression_ratio(dataset_bytes, model_bytes)

        print(f"Размер модели (.pth): {model_bytes} байт")
        print(f"Коэффициент сжатия:   {ratio:.2f}")
        results.append({
            "name": name,
            "kind": "quantized",
            "accuracy": accuracy,
            "bytes": model_bytes,
            "ratio": ratio,
        })
    return results


def run_distillation_experiment(train_loader, test_loader, device, train_dir):
    """Дистилляция.

    Часть 1: 9 учителей (всё, кроме LinearClassifier) → LinearClassifier студент.
             Имена студентов: from_<teacher>.
    Часть 2: 3 conv-учителя → ConvOneLayer студент (для более «честного»
             сравнения внутри свёрточного семейства, иначе линейный студент
             архитектурно не может усвоить conv-знания).
             Имена студентов: conv1_from_<teacher>. ConvOneLayer как учитель
             пропускается — это сама архитектура студента.

    T и α — из constants.py. Сохраняются в saved_models/distilled/.
    Если baseline .pth учителя нет — модель пропускается.
    """
    all_teachers = {
        "one_hidden_layer": OneHiddenLayer,
        "wide_hidden": WideHidden,
        "multi_layer": MultiLayer,
        "tiny_bottleneck": TinyBottleneck,
        "deep_narrow": DeepNarrow,
        "conv_one_layer": ConvOneLayer,
        "conv_two_layers": ConvTwoLayers,
        "conv_deep": ConvDeep,
        "conv_global_pool": ConvGlobalPool,
    }
    # Conv-учители для ConvOneLayer-студента (без самого conv_one_layer).
    conv_teachers_for_conv_student = {
        "conv_two_layers": ConvTwoLayers,
        "conv_deep": ConvDeep,
        "conv_global_pool": ConvGlobalPool,
    }
    DISTILLED_DIR.mkdir(parents=True, exist_ok=True)

    def _distill_one(student_name, student, teacher_name, teacher_cls):
        """Один эпизод дистилляции. Возвращает result-dict или None при пропуске."""
        teacher_path = SAVED_MODELS_DIR / f"{teacher_name}.pth"
        if not teacher_path.exists():
            print(f"\n=== Дистилляция: {student_name} ===  ПРОПУСК (нет {teacher_path.name})")
            return None

        print(f"\n=== Дистилляция: {student_name} ===")
        teacher = teacher_cls()
        teacher.load_state_dict(torch.load(teacher_path, map_location="cpu"))

        student = train_distilled(
            student, teacher, train_loader, device, EPOCHS,
            T=DISTILL_T, alpha=DISTILL_ALPHA, lr=LEARNING_RATE,
        )

        accuracy = evaluate(student, test_loader, device)
        print(f"Точность на тестовой выборке: {accuracy:.4f}")

        student_path = save_model(student, DISTILLED_DIR / f"{student_name}.pth")
        model_bytes = model_size_bytes(student_path)
        dataset_bytes = dataset_size_bytes(train_dir)
        ratio = compression_ratio(dataset_bytes, model_bytes)

        print(f"Размер модели (.pth): {model_bytes} байт")
        print(f"Коэффициент сжатия:   {ratio:.2f}")
        return {
            "name": student_name,
            "kind": "distilled",
            "accuracy": accuracy,
            "bytes": model_bytes,
            "ratio": ratio,
        }

    results = []

    # Часть 1: LinearClassifier-студент от всех 9 учителей
    for teacher_name, teacher_cls in all_teachers.items():
        result = _distill_one(
            f"from_{teacher_name}", LinearClassifier(),
            teacher_name, teacher_cls,
        )
        if result:
            results.append(result)

    # Часть 2: ConvOneLayer-студент от conv-учителей
    for teacher_name, teacher_cls in conv_teachers_for_conv_student.items():
        result = _distill_one(
            f"conv1_from_{teacher_name}", ConvOneLayer(),
            teacher_name, teacher_cls,
        )
        if result:
            results.append(result)

    return results


def _write_results_file(records, path, module_name, header_lines):
    """Перезаписывает один файл результатов, обновляя записи по имени модели."""
    # Уже сохранённые результаты прошлых запусков
    try:
        module = importlib.import_module(module_name)
        importlib.reload(module)
        existing = list(getattr(module, "RESULTS", []))
    except Exception:
        existing = []

    new_records = [
        {
            "name": r["name"],
            "accuracy": r["accuracy"],
            "size_bytes": r["bytes"],
            "ratio": r["ratio"],
        }
        for r in records
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

    lines = list(header_lines) + ["RESULTS = [\n"]
    for rec in all_records:
        lines.append(
            f'    {{"name": "{rec["name"]}", '
            f'"accuracy": {rec["accuracy"]:.4f}, '
            f'"size_bytes": {rec["size_bytes"]}, '
            f'"ratio": {rec["ratio"]:.2f}}},\n'
        )
    lines.append("]\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def save_results(results):
    """Распределяет результаты по файлам в зависимости от поля kind."""
    baseline = [r for r in results if r.get("kind", "baseline") == "baseline"]
    quantized = [r for r in results if r.get("kind") == "quantized"]
    distilled = [r for r in results if r.get("kind") == "distilled"]

    if baseline:
        header = [
            '"""Результаты экспериментов на тестовых выборках.\n',
            "\n",
            "Файл дополняется при запуске execution_scripts/run.py:\n",
            "новые результаты добавляются после уже сохранённых.\n",
            "Каждая запись: имя модели, точность на test, размер .pth в байтах, коэффициент сжатия.\n",
            '"""\n',
            "\n",
        ]
        _write_results_file(baseline, RESULTS_BASELINE_PATH, "results", header)

    if quantized:
        header = [
            '"""Результаты квантизованных моделей (dynamic int8 на Linear-слоях).\n',
            "\n",
            "Файл дополняется при запуске execution_scripts/run.py:\n",
            "новые результаты добавляются после уже сохранённых.\n",
            "Каждая запись: имя модели, точность на test, размер .pth в байтах, коэффициент сжатия.\n",
            '"""\n',
            "\n",
        ]
        _write_results_file(
            quantized, RESULTS_QUANTIZED_PATH, "results_quantized", header
        )

    if distilled:
        header = [
            '"""Результаты дистиллированных моделей.\n',
            "\n",
            "Каждая запись — LinearClassifier-студент, обученный дистилляцией\n",
            "от соответствующего учителя. Имя в формате from_<teacher>.\n",
            "\n",
            "Файл дополняется при запуске execution_scripts/run.py:\n",
            "новые результаты добавляются после уже сохранённых.\n",
            "Каждая запись: имя модели, точность на test, размер .pth в байтах, коэффициент сжатия.\n",
            '"""\n',
            "\n",
        ]
        _write_results_file(
            distilled, RESULTS_DISTILLED_PATH, "results_distilled", header
        )


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
    #results += run_fc_models(train_loader, test_loader, device, train_dir)
    #results += run_conv_models(train_loader, test_loader, device, train_dir)
    #results += run_quantization_experiment(test_loader, train_dir)
    results += run_distillation_experiment(train_loader, test_loader, device, train_dir)

    # Сводка по эксперименту
    print("\n=== Сводка ===")
    for r in results:
        kind = r.get("kind", "baseline")
        suffix = f" [{kind}]" if kind != "baseline" else ""
        print(
            f"{r['name']:>22}{suffix}  точность={r['accuracy']:.4f}  "
            f"размер={r['bytes']} байт  коэффициент={r['ratio']:.2f}"
        )

    # Сохранение результатов: baseline в results.py, квантизованные в
    # results_quantized/, дистиллированные в results_distilled/
    save_results(results)
    print(
        "\nРезультаты сохранены:"
        "\n  - results.py"
        "\n  - results_quantized/__init__.py"
        "\n  - results_distilled/__init__.py"
    )


if __name__ == "__main__":
    main()
