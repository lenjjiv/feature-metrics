# Анализ взаимосвязи признаков

## Обзор

Набор функций для вычисления статистических метрик взаимосвязи между парами признаков в датафрейме. Поддерживаются следующие типы зависимостей:

* Категориальный – категориальный (`cat-cat`)
* Категориальный – числовой (`cat-num`) 
* Числовой – числовой (`num-num`)

## Цель проекта

Предоставить инструмент для расчёта метрик зависимости между признаками и целевой переменной. Метрики выбираются автоматически в зависимости от типа переменных:

### Категориальные признаки (cat-cat)

* Нормализованный коэффициент V (V-Cramer) и p-value (критерий χ²)
* Adjusted Mutual Information (скорректированная взаимная информация)
* Normalized Mutual Information (нормализованная взаимная информация) 
* Adjusted Rand Index

### Категориальный–числовые пары (cat-num)

* F-тест (ANOVA) и p-value
* Коэффициент Point-Biserial с опциями усреднения (`macro`/`micro`/`weighted`) и p-value
* ROC-AUC с усреднением и p-value (через тест Манна–Уитни)
* AUC по Precision-Recall Curve (среднее значение)
* Silhouette Score для оценки разделения групп

### Числовые пары (num-num)

* Pearson correlation, p-value и r²
* Spearman correlation, p-value и r²
* Normalized Mutual Information через `mutual_info_regression`

## Обработка пропущенных значений

Параметр `nan_policy` определяет политику обработки NaN:

* `"drop"`: удаление строк с NaN
* `"raise"`: выброс ошибки при наличии NaN
* `"return_nan"`: возврат NaN при обнаружении NaN в паре
* `"nan_class"`: замена NaN на специальный класс

## Зависимости

```bash
pip install numpy pandas scipy scikit-learn
```

## Примеры использования

### 1. Категориальные признаки (cat-cat)

```python
import pandas as pd
from feature_metrics import calculate_statistics_cat_cat

# Пример данных
data = {
    'feature1': ['A', 'B', 'A', 'C', 'B'],
    'feature2': ['X', 'Y', 'X', 'Z', 'Y'],
    'target':   ['Yes', 'No', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

# Расчет метрик относительно target
result = calculate_statistics_cat_cat(
    df, 
    target='target',
    features=None,
    p_values_only=False,
    quantize_numeric=True,
    nan_policy="drop",
    n_bins=10
)
print(result)

# Попарный расчет метрик
result_dict = calculate_statistics_cat_cat(
    df,
    target=None,
    features=['feature1', 'feature2'],
    p_values_only=False,
    quantize_numeric=True,
    nan_policy="drop",
    n_bins=10
)
```

### 2. Категориальный–числовые пары (cat-num)

```python
from feature_metrics import calculate_statistics_cat_num

data = {
    'num_feature': [1.2, 2.4, 1.8, 3.2, 2.1],
    'cat_feature': ['A', 'B', 'A', 'B', 'A'],
    'target':      ['Yes', 'No', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

# Расчет метрик относительно target
result = calculate_statistics_cat_num(
    df,
    target='target',
    features=None,
    p_values_only=False,
    nan_policy="drop",
    averaging="macro"
)
print(result)
```

### 3. Числовые признаки (num-num)

```python
from feature_metrics import calculate_statistics_num_num

data = {
    'num_feature1': [1.2, 2.4, 1.8, 3.2, 2.1],
    'num_feature2': [4.5, 3.2, 4.8, 2.9, 3.7],
    'target':       [0.5, 1.1, 0.7, 1.5, 1.0]
}
df = pd.DataFrame(data)

# Расчет метрик относительно target
result = calculate_statistics_num_num(
    df,
    target='target',
    features=None,
    p_values_only=False,
    nan_policy="drop"
)
print(result)
```

## API Reference

### Общие параметры

* `df` (pandas.DataFrame): Исходный датафрейм
* `target` (None, str, pandas.Series): Целевая переменная
* `features` (list, None): Список признаков для расчета
* `p_values_only` (bool): Возврат только p-value
* `nan_policy` (str): Политика обработки NaN

### Специфические параметры

#### calculate_statistics_cat_cat
* `quantize_numeric` (bool): Квантизация числовых признаков
* `n_bins` (int): Количество бинов для квантизации

#### calculate_statistics_cat_num
* `averaging` (str): Метод усреднения (`"macro"`, `"weighted"`, `"micro"`)

### Возвращаемые значения

* При `target ≠ None`: pandas.DataFrame с метриками по признакам
* При `target = None`: словарь матриц попарных метрик

## Детали реализации метрик

### Категориальные признаки
* V-Cramer: нормализованный χ² с учетом размеров таблицы
* Adjusted/Normalized Mutual Information: корректировка на размер выборки
* Adjusted Rand Index: оценка согласованности классификаций

### Категориальный–числовые пары
* F-тест: сравнение дисперсий между группами
* Point-Biserial: корреляция для бинарных/мультиклассовых случаев
* ROC-AUC и PR-AUC: оценки качества разделения
* Silhouette Score: оценка кластеризации

### Числовые признаки
* Корреляции Пирсона/Спирмена с квадратами
* Normalized Mutual Information через регрессию
