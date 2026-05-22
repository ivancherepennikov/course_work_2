import torch
import torch.nn as nn


def train(model, train_loader, device, epochs, lr=1e-3):
    """Цикл обучения модели.

    model        — обучаемая модель
    train_loader — DataLoader с обучающей выборкой
    device       — устройство (mps / cpu)
    epochs       — количество эпох
    lr           — learning rate

    За каждую эпоху печатает средний loss и точность на train.
    Возвращает обученную модель.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total
        print(f"Эпоха {epoch + 1}/{epochs}  loss={avg_loss:.4f}  accuracy={accuracy:.4f}")

    return model
