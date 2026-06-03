"""Дистилляция знаний (knowledge distillation).

Студент учится повторять поведение учителя через комбинированный лосс:

    L = α · CE(student_logits, hard_labels)
      + (1 − α) · T² · KL(softmax(student/T) || softmax(teacher/T))

  T (температура) — большее T размазывает softmax учителя, открывая
                    студенту структуру относительных вероятностей
                    («dark knowledge»).
  α                — вес хардовых меток. α=1 — обычное обучение,
                    α=0 — студент полностью копирует распределение учителя.
  T²               — множитель, компенсирующий 1/T² в градиентах softmax.

Учитель должен быть уже обучен; в цикле он в eval-режиме и без градиентов.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def train_distilled(student, teacher, train_loader, device, epochs, T, alpha, lr=1e-3):
    """Обучает студента дистилляцией от учителя.

    student      — необученная (свежая) модель-студент
    teacher      — обученная модель-учитель
    train_loader — DataLoader с обучающей выборкой
    device       — устройство (mps / cpu)
    epochs       — количество эпох
    T            — температура softmax
    alpha        — вес хардовых меток (от 0 до 1)
    lr           — learning rate

    Возвращает обученного студента. Учитель не модифицируется.
    """
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()

    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)

            hard = ce(student_logits, labels)
            soft = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction="batchmean",
            ) * (T * T)
            loss = alpha * hard + (1 - alpha) * soft

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = student_logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total
        print(f"Эпоха {epoch + 1}/{epochs}  loss={avg_loss:.4f}  accuracy={accuracy:.4f}")

    return student
