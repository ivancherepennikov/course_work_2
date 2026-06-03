"""Квантизация моделей.

Сейчас реализована только dynamic int8 квантизация — самая простая:
веса nn.Linear переводятся из float32 в int8, активации остаются float
и квантуются на лету. Свёрточные слои не трогаются.

Квантизованные модели работают только на CPU (MPS не поддерживает int8-ядра).
"""

import torch
import torch.nn as nn


def quantize_dynamic(model):
    """Применяет dynamic int8 квантизацию к Linear-слоям модели.

    Возвращает новую (квантизованную) модель. Исходная не меняется.
    """
    # На Apple Silicon квантизованный бэкенд по умолчанию не выбран —
    # ставим qnnpack (единственный поддерживаемый на этой сборке PyTorch).
    torch.backends.quantized.engine = "qnnpack"
    model.eval()
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
