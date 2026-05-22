import torch.nn as nn


class LinearClassifier(nn.Module):
    """Самая простая модель: один линейный слой без скрытых слоёв.

    Вход  — изображение 28x28 (784 признака после разворачивания).
    Выход — 10 логитов (по числу классов MNIST).
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)


class OneHiddenLayer(nn.Module):
    """Полносвязная модель с одним скрытым слоём на 10 нейронов.

    Вход   — изображение 28x28 (784 признака после разворачивания).
    Скрытый слой — 10 нейронов с активацией ReLU.
    Выход  — 10 логитов (по числу классов MNIST).
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28 * 28, 10)
        self.relu = nn.ReLU()
        self.output = nn.Linear(10, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.hidden(x))
        return self.output(x)


class WideHidden(nn.Module):
    """Полносвязная модель с широким скрытым слоём: 784 -> 784 -> 10."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 784),
            nn.ReLU(),
            nn.Linear(784, 10),
        )

    def forward(self, x):
        return self.net(x)


class MultiLayer(nn.Module):
    """Многослойная полносвязная модель: 784 -> 64 -> 32 -> 16 -> 10."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
        )

    def forward(self, x):
        return self.net(x)


class TinyBottleneck(nn.Module):
    """Крошечная модель с узким бутылочным горлом: 784 -> 8 -> 10."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 8),
            nn.ReLU(),
            nn.Linear(8, 10),
        )

    def forward(self, x):
        return self.net(x)


class DeepNarrow(nn.Module):
    """Глубокая узкая модель: 784 -> 32 -> 32 -> 32 -> 10."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.net(x)
