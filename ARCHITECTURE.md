# Архитектура проекта

## Тема

Поиск оптимальных коэффициентов сжатия для моделей машинного обучения.

**Цель:** найти модели с максимальным **коэффициентом сжатия**, при которых
точность на тестовой выборке не падает ниже заданного порога.

**Коэффициент сжатия:**

```
K = размер_датасета (байты) / размер_модели (байты)
```

где размер модели — это размер сохранённого `.pth` файла на диске,
а размер датасета — суммарный размер всех файлов обучающей выборки.

## Используемые методы сжатия

В проекте реализованы и сравнены три подхода:

1. **Выбор архитектуры (baseline)** — обучение 10 разных моделей с нуля
   (от линейной до глубокой свёрточной). Сравнение их «исходного» коэффициента
   сжатия и точности.
2. **Динамическая int8 квантизация** — перевод весов `nn.Linear`-слоёв
   из float32 в int8 через `torch.quantization.quantize_dynamic`.
3. **Дистилляция знаний (knowledge distillation)** — обучение маленького
   студента с использованием выходного распределения уже обученного учителя.
   Loss: `α·CE(student, hard_labels) + (1−α)·T²·KL(student/T || teacher/T)`.

Прунинг рассмотрен, но не реализован: модели слишком малы для структурного
прунинга, а unstructured pruning не уменьшает `.pth` без отдельной обработки.

## Датасет

**MNIST** (60 000 train + 10 000 test, 28×28 grayscale). Скачан через
torchvision и сохранён локально в виде PNG-файлов в раскладке `ImageFolder`:

```
datasets/data/mnist_images/
├── train/<class>/<index>.png
└── test/<class>/<index>.png
```

В числитель коэффициента сжатия идёт размер папки `train/` в байтах
(16 195 658 байт ≈ 15.45 МиБ).

## Структура директорий

```
course_work_2/
├── course_work_venv/                виртуальное окружение
├── requirements.txt                 зависимости
├── ARCHITECTURE.md                  этот файл
├── THESIS_CONTEXT.md                полный технический контекст для текста курсовой
├── test.py                          черновик (не пушится в git)
├── .gitignore
├── constants.py                     гиперпараметры: EPOCHS, DISTILL_T, DISTILL_ALPHA
│
├── models/                          определения моделей
│   ├── fc.py                        6 полносвязных архитектур
│   └── cnn.py                       4 свёрточные архитектуры
│
├── core/                            общий код
│   ├── train.py                     цикл обучения (Adam + CrossEntropy)
│   ├── inference.py                 расчёт точности на test
│   ├── sizes.py                     измерение размеров в байтах, коэффициент
│   ├── quantization.py              dynamic int8 квантизация Linear-слоёв
│   └── distillation.py              цикл обучения студента через distillation
│
├── datasets/
│   └── data/mnist_images/           MNIST в формате PNG (ImageFolder)
│
├── saved_models/                    обученные веса
│   ├── <name>.pth                   10 baseline моделей
│   ├── quantized/                   10 квантизованных моделей
│   └── distilled/                   12 дистиллированных студентов
│
├── results.py                       результаты baseline моделей
├── results_quantized/
│   └── __init__.py                  результаты квантизованных моделей
├── results_distilled/
│   └── __init__.py                  результаты дистиллированных моделей
│
└── execution_scripts/               точки входа
    ├── run.py                       главный скрипт: обучение, квантизация, дистилляция
    ├── create_score_table.py        tkinter-таблица результатов (3 вкладки)
    └── draw_graph.py                Pareto-диаграмма точность ↔ K (matplotlib)
```

## Ответственность модулей

### `models/`
- `fc.py` — `LinearClassifier`, `OneHiddenLayer`, `WideHidden`, `MultiLayer`,
  `TinyBottleneck`, `DeepNarrow`.
- `cnn.py` — `ConvOneLayer`, `ConvTwoLayers`, `ConvDeep`, `ConvGlobalPool`.

### `core/`
- `train.py` — `train(model, train_loader, device, epochs, lr)`. Adam + CE,
  печатает loss и точность на train по эпохам.
- `inference.py` — `evaluate(model, test_loader, device)`. Возвращает долю
  верных предсказаний на test.
- `sizes.py` — `save_model`, `model_size_bytes`, `dataset_size_bytes`,
  `compression_ratio`.
- `quantization.py` — `quantize_dynamic(model)`. Внутри устанавливает
  `qnnpack` бэкенд для Apple Silicon и применяет
  `torch.quantization.quantize_dynamic` к `nn.Linear` слоям.
- `distillation.py` — `train_distilled(student, teacher, ..., T, alpha, lr)`.
  Комбинированный лосс на soft и hard метках.

### `execution_scripts/`
- `run.py` — главный скрипт. Объединяет четыре эксперимента:
  - `run_fc_models()` — обучает 6 FC моделей с нуля
  - `run_conv_models()` — обучает 4 conv модели с нуля
  - `run_quantization_experiment()` — квантизует все 10 моделей
  - `run_distillation_experiment()` — две части:
    - 9 учителей → LinearClassifier-студент (`from_<teacher>`)
    - 3 conv-учителя → ConvOneLayer-студент (`conv1_from_<teacher>`)

  В `main()` вызовы каждой функции в `results += ...` строке —
  можно закомментировать ненужное, чтобы не пересчитывать.

- `create_score_table.py` — интерактивная таблица (tkinter Treeview) с
  тремя вкладками: «Все модели», «Квантизация», «Дистилляция». Сортировка
  по клику на заголовок колонки.

- `draw_graph.py` — двухмерная диаграмма «точность ↔ K» (matplotlib).
  Все точки маркерами по категориям; **Парето-фронт** выделен чёрными
  обводками и подписями.

### Файлы результатов
Все три файла имеют одинаковый формат: модуль с переменной `RESULTS` —
список словарей `{"name", "accuracy", "size_bytes", "ratio"}`. Записи
дописываются и обновляются по имени модели при каждом запуске `run.py`.

## Гиперпараметры (`constants.py`)

| Параметр | Значение | Назначение |
|---|---|---|
| `EPOCHS` | 10 | количество эпох обучения |
| `DISTILL_T` | 3 | температура softmax в distillation |
| `DISTILL_ALPHA` | 0.75 | вес хардовых меток (1−α — вес soft-labels учителя) |

`BATCH_SIZE = 64` и `LEARNING_RATE = 1e-3` заданы в `run.py` как константы
модуля.

## Технический стек

- **Python** 3.11
- **PyTorch** 2.12 — обучение, квантизация
- **torchvision** 0.27 — загрузка MNIST, ImageFolder
- **matplotlib** 3.10 — Pareto-диаграмма
- **tkinter** (stdlib) — интерактивная таблица результатов
- **Hardware:** Apple Silicon с MPS-ускорителем для обучения;
  квантизованный инференс выполняется на CPU через бэкенд `qnnpack`.

## Команды запуска

```bash
# Активировать виртуалку (или указывать ./course_work_venv/bin/python явно)
source course_work_venv/bin/activate

# Главный эксперимент (обучение / квантизация / дистилляция)
python execution_scripts/run.py

# Таблица результатов с тремя вкладками
python execution_scripts/create_score_table.py

# Pareto-диаграмма «точность ↔ K»
python execution_scripts/draw_graph.py
```

`run.py` обновляет файлы `results.py`, `results_quantized/__init__.py`,
`results_distilled/__init__.py`; обе утилиты визуализации читают из них.
