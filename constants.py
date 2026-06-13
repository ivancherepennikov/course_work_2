# Количество эпох обучения
EPOCHS = 10

# Гиперпараметры дистилляции (knowledge distillation)
DISTILL_T = 3          # температура softmax
DISTILL_ALPHA = 0.75   # вес хардовых меток (высокий → больше доверяем меткам)
