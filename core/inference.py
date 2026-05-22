import torch


def evaluate(model, test_loader, device):
    """Прогон модели на тестовой выборке.

    model       — обученная модель
    test_loader — DataLoader с тестовой выборкой
    device      — устройство (mps / cpu)

    Возвращает точность — долю верных предсказаний.
    """
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total
