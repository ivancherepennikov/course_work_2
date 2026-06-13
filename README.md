# Поиск оптимальных коэффициентов сжатия для моделей машинного обучения

Курсовая работа: исследование методов сжатия нейронных сетей на задаче
классификации рукописных цифр MNIST.

**Цель:** найти модели с максимальным **коэффициентом сжатия**
`K = размер_датасета / размер_модели`, при которых точность на тестовой
выборке остаётся выше заданного порога.

## О работе

Сравниваются три метода уменьшения размера модели на одной задаче:

1. **Выбор архитектуры (baseline)** — обучение 10 моделей разной capacity
   с нуля (6 полносвязных + 4 свёрточные).
2. **Динамическая int8 квантизация** — перевод весов `nn.Linear`-слоёв
   из `float32` в `int8` через `torch.quantization.quantize_dynamic`.
3. **Дистилляция знаний (Hinton-style)** — обучение маленького студента
   под комбинированным лоссом с использованием soft labels учителя
   (`T = 3`, `α = 0.75`). Два варианта студентов:
   - `LinearClassifier` для всех 9 учителей (`from_<teacher>`)
   - `ConvOneLayer` для 3 свёрточных учителей (`conv1_from_<teacher>`)

Прунинг рассмотрен в обзоре методов, но не реализован: для моделей такого
размера структурный прунинг даёт почти пустые сети, а unstructured pruning
не уменьшает `.pth` без отдельной обработки.

Всего — **32 экспериментальные конфигурации**, объединённые на одном
Парето-фронте «точность ↔ коэффициент сжатия».

## Ключевой результат

Из 32 экспериментов на **Парето-фронте — 6 моделей**, из которых
5 — квантизованные. Ни одна дистиллированная модель не достигла фронта.

| Модель | Точность | K | Категория |
|---|---|---|---|
| `conv_two_layers` | 0.9901 | 189.86 | baseline |
| `conv_two_layers` | 0.9900 | 390.02 | quantized |
| `conv_one_layer` | 0.9793 | 763.26 | quantized |
| `one_hidden_layer` | 0.9324 | 1305.36 | quantized |
| `linear_classifier` | 0.9281 | 1439.23 | quantized |
| `tiny_bottleneck` | 0.9161 | 1465.27 | quantized |

Вывод: квантизация — рабочий и почти «бесплатный» метод сжатия в 2.5–4 раза
без потери точности. Дистилляция в нашем сетапе оказалась хуже квантизации —
подробный разбор причин в тексте курсовой.

## Технологический стек

- **Python** 3.11
- **PyTorch** 2.12 (с MPS-ускорителем на Apple Silicon)
- **torchvision** 0.27 — загрузка MNIST
- **matplotlib** 3.10 — Парето-диаграмма
- **tkinter** (стандартная библиотека) — интерактивная таблица результатов
- Квантизованный инференс — через CPU-бэкенд `qnnpack`

## Установка

```bash
git clone https://github.com/ivancherepennikov/course_work_2.git
cd course_work_2

# Виртуальное окружение
python3.11 -m venv course_work_venv
source course_work_venv/bin/activate

# Зависимости
pip install -r requirements.txt
```

### Подготовка датасета MNIST

При первом запуске нужно один раз сгенерировать PNG-файлы из дистрибутива
torchvision (это формат, в котором проект ожидает данные):

```bash
python -c "
from torchvision import datasets
from pathlib import Path

root = 'datasets/data'
out = Path('datasets/data/mnist_images')

datasets.MNIST(root=root, train=True, download=True)
datasets.MNIST(root=root, train=False, download=True)

for split, flag in [('train', True), ('test', False)]:
    ds = datasets.MNIST(root=root, train=flag, download=False)
    for cls in range(10):
        (out / split / str(cls)).mkdir(parents=True, exist_ok=True)
    for idx, (img, label) in enumerate(ds):
        img.save(out / split / str(label) / f'{idx:05d}.png')
print('MNIST подготовлен')
"
```

Скрипт скачивает MNIST и распаковывает 70 000 PNG-файлов в раскладке
`datasets/data/mnist_images/{train,test}/<class>/`. Занимает ~1 минуту.

## Как запускать скрипты

Три точки входа в папке `execution_scripts/`.

### 1. Главный эксперимент — `run.py`

Объединяет четыре стадии:
- обучение 6 FC моделей с нуля
- обучение 4 conv моделей с нуля
- квантизация всех 10 моделей
- дистилляция (12 студентов)

```bash
python execution_scripts/run.py
```

В `main()` каждая стадия — отдельная строка `results += run_*(...)`.
Ненужные строки можно **закомментировать**, чтобы не пересчитывать.
Например, если baseline-модели уже обучены и `.pth` лежат в `saved_models/`,
оставь только квантизацию и дистилляцию.

После прогона обновятся файлы:
- `results.py` — точность baseline моделей
- `results_quantized/__init__.py` — точность квантизованных
- `results_distilled/__init__.py` — точность дистиллированных

И сохранятся `.pth`-чекпоинты в `saved_models/`, `saved_models/quantized/`,
`saved_models/distilled/`.

Полный прогон на MPS — ориентировочно 30–60 минут.

### 2. Таблица результатов — `create_score_table.py`

Открывает интерактивное окно (tkinter Treeview) с тремя вкладками:
- **«Все модели»** — 10 baseline моделей с архитектурой, точностью, K
- **«Квантизация»** — для каждой квантизованной модели: точность,
  изменение точности и коэффициент сжатия
- **«Дистилляция»** — то же для 12 дистиллированных студентов

Сортировка — клик на заголовок колонки.

```bash
python execution_scripts/create_score_table.py
```

### 3. Pareto-диаграмма — `draw_graph.py`

Двухмерный график «точность ↔ K» через matplotlib. Все 32 точки на одном
плоте, **Парето-фронт выделен** чёрными обводками с подписями имён
моделей; внутри-категорийные точки — полупрозрачными маркерами.

```bash
python execution_scripts/draw_graph.py
```

В нижней панели окна matplotlib есть иконка сохранения — можно выгрузить
PNG/PDF/SVG для отчёта.

## Структура проекта

```
course_work_2/
├── constants.py                     EPOCHS, DISTILL_T, DISTILL_ALPHA
├── requirements.txt
├── ARCHITECTURE.md                  подробное описание архитектуры
├── THESIS_CONTEXT.md                технический контекст для текста курсовой
│
├── models/                          определения моделей
│   ├── fc.py                        6 полносвязных архитектур
│   └── cnn.py                       4 свёрточные архитектуры
│
├── core/                            общий код
│   ├── train.py                     цикл обучения
│   ├── inference.py                 расчёт точности на test
│   ├── sizes.py                     размеры файлов, коэффициент сжатия
│   ├── quantization.py              dynamic int8 квантизация
│   └── distillation.py              обучение через distillation
│
├── datasets/data/mnist_images/      MNIST в формате PNG
│
├── saved_models/                    обученные веса
│   ├── *.pth                        10 baseline моделей
│   ├── quantized/                   10 квантизованных
│   └── distilled/                   12 дистиллированных студентов
│
├── results.py                       результаты baseline
├── results_quantized/__init__.py    результаты квантизации
├── results_distilled/__init__.py    результаты дистилляции
│
└── execution_scripts/
    ├── run.py                       главный эксперимент
    ├── create_score_table.py        таблица результатов (tkinter)
    └── draw_graph.py                Pareto-диаграмма (matplotlib)
```

Подробнее: [ARCHITECTURE.md](ARCHITECTURE.md).

## Гиперпараметры

Заданы в `constants.py`:

| Параметр | Значение | Назначение |
|---|---|---|
| `EPOCHS` | 10 | количество эпох обучения |
| `DISTILL_T` | 3 | температура softmax в дистилляции |
| `DISTILL_ALPHA` | 0.75 | вес хардовых меток (1 − α — вес soft labels учителя) |

`BATCH_SIZE = 64` и `LEARNING_RATE = 1e-3` — константы внутри `run.py`.

## Автор

Иван Черепенников, НИУ ВШЭ.
