import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn.metrics import silhouette_score, roc_auc_score, average_precision_score
from sklearn.feature_selection import mutual_info_regression


# Вспомогательная функция для обработки NaN в паре признаков
def handle_nan(x, y, nan_policy):
    """
    Обрабатывает NaN согласно выбранной политике.

    nan_policy:
      - "drop": удаляет строки с NaN;
      - "raise": выбрасывает ошибку при наличии NaN;
      - "return_nan": если есть NaN, то метрика возвращается как NaN;
      - "nan_class": заменяет NaN на специальный класс.
    """
    if nan_policy == "drop":
        mask = (~x.isna()) & (~y.isna())
        return x[mask], y[mask], False
    elif nan_policy == "raise":
        if x.isna().any() or y.isna().any():
            raise ValueError("Обнаружены NaN значения в данных")
        return x, y, False
    elif nan_policy == "return_nan":
        if x.isna().any() or y.isna().any():
            return x, y, True
        return x, y, False
    elif nan_policy == "nan_class":
        marker_x = "__nan__"
        while marker_x in set(x.dropna().unique()):
            marker_x += "_"
        x = x.fillna(marker_x)
        marker_y = "__nan__"
        while marker_y in set(y.dropna().unique()):
            marker_y += "_"
        y = y.fillna(marker_y)
        return x, y, False
    else:
        raise ValueError("Неизвестная политика NaN: " + str(nan_policy))


# 1. Функция для категориальных признаков (cat-cat)
def compute_cat_cat_metrics(x, y, nan_policy):
    """
    Вычисляет следующие метрики для пары категориальных признаков:
      - Нормализованный коэффициент V (V-Cramer) и p-value (по критерию χ²)
      - Adjusted Mutual Information
      - Normalized Mutual Information
      - Adjusted Rand Index
    """
    x, y, return_nan_flag = handle_nan(x, y, nan_policy)
    if return_nan_flag:
        return {
            "v_cramer": np.nan,
            "v_cramer_p": np.nan,
            "adjusted_mutual_info": np.nan,
            "normalized_mutual_info": np.nan,
            "adjusted_rand_index": np.nan,
        }
    # Если один из признаков константен, метрики не определены
    if x.nunique() < 2 or y.nunique() < 2:
        return {
            "v_cramer": np.nan,
            "v_cramer_p": np.nan,
            "adjusted_mutual_info": np.nan,
            "normalized_mutual_info": np.nan,
            "adjusted_rand_index": np.nan,
        }
    contingency = pd.crosstab(x, y)
    try:
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
    except Exception:
        chi2, p = np.nan, np.nan
    n = contingency.to_numpy().sum()
    min_dim = min(contingency.shape) - 1
    if min_dim <= 0:
        v_cramer = np.nan
    else:
        v_cramer = np.sqrt(chi2 / (n * min_dim))
    ami = metrics.adjusted_mutual_info_score(x, y)
    nmi = metrics.normalized_mutual_info_score(x, y)
    ari = metrics.adjusted_rand_score(x, y)
    return {
        "v_cramer": v_cramer,
        "v_cramer_p": p,
        "adjusted_mutual_info": ami,
        "normalized_mutual_info": nmi,
        "adjusted_rand_index": ari,
    }


# 2. Функция для пар категориальный–числовой (cat-num)
def compute_cat_num_metrics(num_series, cat_series, nan_policy, averaging="macro"):
    """
    Вычисляет метрики для пары (числовой признак, категориальный признак):
      - F-тест (ANOVA) и p-value;
      - Коэффициент Point-Biserial (для бинарного случая, а для мультикласса – усреднение по опциям macro/micro/weighted) и p-value;
      - ROC-AUC (усреднение по классам) и p-value (p-value считается через Mann–Whitney U);
      - AUC по Precision-Recall Curve (среднее значение);
      - Silhouette Score.
    """
    num_series, cat_series, return_nan_flag = handle_nan(
        num_series, cat_series, nan_policy
    )
    if return_nan_flag:
        return {
            "f_test": np.nan,
            "f_test_p": np.nan,
            "pointbiserial": np.nan,
            "pointbiserial_p": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "silhouette": np.nan,
        }
    if cat_series.nunique() < 2 or num_series.nunique() < 2:
        return {
            "f_test": np.nan,
            "f_test_p": np.nan,
            "pointbiserial": np.nan,
            "pointbiserial_p": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "silhouette": np.nan,
        }
    # F-тест (ANOVA)
    groups = [group.dropna() for _, group in num_series.groupby(cat_series)]
    if len(groups) < 2:
        f_stat, f_p = np.nan, np.nan
    else:
        try:
            f_stat, f_p = stats.f_oneway(*groups)
        except Exception:
            f_stat, f_p = np.nan, np.nan

    # Point-Biserial
    if cat_series.nunique() == 2:
        mapping = {val: idx for idx, val in enumerate(sorted(cat_series.unique()))}
        binary = cat_series.map(mapping)
        try:
            pb_r, pb_p = stats.pointbiserialr(num_series, binary)
        except Exception:
            pb_r, pb_p = np.nan, np.nan
    else:
        r_list = []
        p_list = []
        weights = []
        for val in cat_series.unique():
            binary = (cat_series == val).astype(int)
            if binary.nunique() < 2:
                continue
            try:
                r_val, p_val = stats.pointbiserialr(num_series, binary)
                r_list.append(r_val)
                p_list.append(p_val)
                weights.append(binary.sum())
            except Exception:
                continue
        if len(r_list) == 0:
            pb_r, pb_p = np.nan, np.nan
        else:
            if averaging == "macro":
                pb_r = np.mean(r_list)
            elif averaging in ["weighted", "micro"]:
                pb_r = np.average(r_list, weights=weights)
            else:
                pb_r = np.mean(r_list)
            try:
                _, pb_p = stats.combine_pvalues(p_list)
            except Exception:
                pb_p = np.nan

    # ROC-AUC
    auc_list = []
    weights_auc = []
    for val in cat_series.unique():
        binary = (cat_series == val).astype(int)
        if binary.nunique() < 2:
            continue
        try:
            auc = roc_auc_score(binary, num_series) * 2 - 1
        except Exception:
            auc = np.nan
        auc_list.append(auc)
        weights_auc.append(binary.sum())
    if len(auc_list) == 0:
        roc_auc_avg = np.nan
    else:
        if averaging == "macro":
            roc_auc_avg = np.mean(auc_list)
        elif averaging in ["weighted", "micro"]:
            roc_auc_avg = np.average(auc_list, weights=weights_auc)
        else:
            roc_auc_avg = np.mean(auc_list)

    # Precision-Recall AUC (среднее значение average precision)
    pr_list = []
    weights_pr = []
    for val in cat_series.unique():
        binary = (cat_series == val).astype(int)
        if binary.nunique() < 2:
            continue
        try:
            pr = average_precision_score(binary, num_series)
        except Exception:
            pr = np.nan
        pr_list.append(pr)
        weights_pr.append(binary.sum())
    if len(pr_list) == 0:
        pr_auc_avg = np.nan
    else:
        if averaging == "macro":
            pr_auc_avg = np.mean(pr_list)
        elif averaging in ["weighted", "micro"]:
            pr_auc_avg = np.average(pr_list, weights=weights_pr)
        else:
            pr_auc_avg = np.mean(pr_list)

    # Silhouette Score (для оценки разделения групп)
    try:
        if cat_series.nunique() < 2:
            sil = np.nan
        else:
            sil = silhouette_score(num_series.values.reshape(-1, 1), cat_series)
    except Exception:
        sil = np.nan

    return {
        "f_test": f_stat,
        "f_test_p": f_p,
        "pointbiserial": pb_r,
        "pointbiserial_p": pb_p,
        "roc_auc": roc_auc_avg,
        "pr_auc": pr_auc_avg,
        "silhouette": sil,
    }


# 3. Функция для числовых признаков (num-num)
def compute_num_num_metrics(x, y, nan_policy):
    """
    Вычисляет следующие метрики для пары числовых признаков:
      - Pearson correlation, его p-value и квадрат коэффициента (r²);
      - Spearman correlation, его p-value и квадрат коэффициента;
      - Normalized Mutual Information (без p-value).
    """
    x, y, return_nan_flag = handle_nan(x, y, nan_policy)
    if return_nan_flag or len(x) < 2:
        return {
            "pearson": np.nan,
            "pearson_p": np.nan,
            "pearson_r2": np.nan,
            "spearman": np.nan,
            "spearman_p": np.nan,
            "spearman_r2": np.nan,
            "mutual_info": np.nan,
        }
    try:
        pearson_r, pearson_p = stats.pearsonr(x, y)
    except Exception:
        pearson_r, pearson_p = np.nan, np.nan
    try:
        spearman_r, spearman_p = stats.spearmanr(x, y)
    except Exception:
        spearman_r, spearman_p = np.nan, np.nan

    # Вычисляем mutual information и нормализуем
    try:
        # Считаем MI только один раз для каждой переменной
        mi_x = mutual_info_regression(
            x.values.reshape(-1, 1), x.values, random_state=0
        )[0]
        mi_y = mutual_info_regression(
            y.values.reshape(-1, 1), y.values, random_state=0
        )[0]
        mi_xy = mutual_info_regression(
            x.values.reshape(-1, 1), y.values, random_state=0
        )[0]

        # Проверяем корректность значений и нормализуем
        if not (
            np.isfinite(mi_x) and np.isfinite(mi_y) and mi_x > 1e-10 and mi_y > 1e-10
        ):  # используем малое число вместо точного нуля
            normalized_mi = np.nan
        else:
            norm_factor = np.sqrt(mi_x * mi_y)
            normalized_mi = mi_xy / norm_factor
    except Exception:
        normalized_mi = np.nan

    return {
        "pearson": pearson_r,
        "pearson_p": pearson_p,
        "pearson_r2": pearson_r**2 if not np.isnan(pearson_r) else np.nan,
        "spearman": spearman_r,
        "spearman_p": spearman_p,
        "spearman_r2": spearman_r**2 if not np.isnan(spearman_r) else np.nan,
        "mutual_info": normalized_mi,
    }


# Функция для расчёта метрик для пары категориальных признаков (cat-cat)
def calculate_statistics_cat_cat(
    df,
    target,
    features=None,
    p_values_only=False,
    quantize_numeric=True,
    nan_policy="drop",
    n_bins=10,
):
    """
    Если target задан (как строка или pd.Series), то рассчитываются метрики между каждым выбранным признаком и таргетом.
    Если target=None, то рассчитываются попарно метрики между признаками (с использованием только категориальных признаков,
    либо с квантизацией числовых, если quantize_numeric=True).
    """
    df = df.copy()
    if target is not None:
        # Если target задан, извлекаем его
        if isinstance(target, str):
            target_series = df.pop(target)
        elif isinstance(target, pd.Series):
            target_series = target
        else:
            raise ValueError("target должен быть None, str или pd.Series")
        # Определяем список признаков: если features=None, берём все колонки, кроме таргета
        if features is None:
            features = df.columns.tolist()
        else:
            features = list(features)
        results = {}
        for col in features:
            series = df[col]
            # Если включена квантизация и признак числовой, то преобразуем его в категориальный с помощью qcut
            if quantize_numeric and pd.api.types.is_numeric_dtype(series):
                try:
                    series = pd.qcut(series, q=n_bins, duplicates="drop")
                except Exception:
                    series = series.astype(str)
            else:
                # Если квантизация не включена, а признак числовой – пропускаем его
                if pd.api.types.is_numeric_dtype(series):
                    continue
            results[col] = compute_cat_cat_metrics(series, target_series, nan_policy)
        result_df = pd.DataFrame(results).T
        if p_values_only:
            result_df = result_df[
                [col for col in result_df.columns if col.endswith("_p")]
            ]
        return result_df
    else:
        # target=None: рассчитываем попарно метрики
        if features is None:
            features = df.columns.tolist()
        else:
            features = list(features)
        processed = {}
        for col in features:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                if quantize_numeric:
                    try:
                        series = pd.qcut(series, q=n_bins, duplicates="drop")
                    except Exception:
                        series = series.astype(str)
                else:
                    continue
            processed[col] = series
        feature_names = list(processed.keys())
        metric_keys = [
            "v_cramer",
            "v_cramer_p",
            "adjusted_mutual_info",
            "normalized_mutual_info",
            "adjusted_rand_index",
        ]
        # Инициализируем матрицы результатов для каждой метрики
        results = {
            key: pd.DataFrame(index=feature_names, columns=feature_names)
            for key in metric_keys
        }
        for i, col1 in enumerate(feature_names):
            for j in range(i, len(feature_names)):
                col2 = feature_names[j]
                met_dict = compute_cat_cat_metrics(
                    processed[col1], processed[col2], nan_policy
                )
                for key in metric_keys:
                    results[key].at[col1, col2] = met_dict[key]
                    results[key].at[col2, col1] = met_dict[key]
        if p_values_only:
            results = {key: df for key, df in results.items() if key.endswith("_p")}
        return results


# Функция для расчёта метрик для пары категориальный–числовой (cat-num)
def calculate_statistics_cat_num(
    df, target, features=None, p_values_only=False, nan_policy="drop", averaging="micro"
):
    """
    Если target задан, то:
      - если таргет числовой, то рассчитываются метрики между таргетом (числовой) и категориальными признаками из df;
      - если таргет категориальный, то между числовыми признаками из df и таргетом.
    Если target=None, то рассчитываются метрики для попарного сравнения числовых и категориальных признаков.

    Столбцы итогового датафрейма: строка – числовой признак, столбец – категориальный.
    """
    df = df.copy()
    if target is not None:
        if isinstance(target, str):
            target_series = df.pop(target)
        elif isinstance(target, pd.Series):
            target_series = target
        else:
            raise ValueError("target должен быть None, str или pd.Series")
        # Если таргет числовой, то рассматриваем его как числовой и признаки df должны быть категориальными
        if pd.api.types.is_numeric_dtype(target_series):
            cat_features = [
                col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])
            ]
            results = {}
            for col in cat_features:
                results[col] = compute_cat_num_metrics(
                    target_series, df[col], nan_policy, averaging
                )
            result_df = pd.DataFrame(results).T
        else:
            # Таргет категориальный: числовые признаки из df
            num_features = [
                col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
            ]
            results = {}
            for col in num_features:
                results[col] = compute_cat_num_metrics(
                    df[col], target_series, nan_policy, averaging
                )
            result_df = pd.DataFrame(results).T
        if p_values_only:
            # Оставляем только колонки, содержащие p-value (например, f_test_p, pointbiserial_p)
            result_df = result_df[
                [
                    col
                    for col in result_df.columns
                    if col.endswith("_p") or col == "f_test"
                ]
            ]
        return result_df
    else:
        # target=None: попарное сравнение числовых и категориальных признаков из df
        num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        cat_cols = [
            col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])
        ]
        metric_keys = [
            "f_test",
            "f_test_p",
            "pointbiserial",
            "pointbiserial_p",
            "roc_auc",
            "pr_auc",
            "silhouette",
        ]
        results = {
            key: pd.DataFrame(index=num_cols, columns=cat_cols) for key in metric_keys
        }
        for num_col in num_cols:
            for cat_col in cat_cols:
                met_dict = compute_cat_num_metrics(
                    df[num_col], df[cat_col], nan_policy, averaging
                )
                for key in metric_keys:
                    results[key].at[num_col, cat_col] = met_dict[key]
        if p_values_only:
            results = {
                key: df
                for key, df in results.items()
                if key.endswith("_p") or key == "f_test"
            }
        return results


# Функция для расчёта метрик для пары числовых признаков (num-num)
def calculate_statistics_num_num(
    df, target, features=None, p_values_only=False, nan_policy="drop"
):
    """
    Если target задан, то рассчитываются метрики между каждым числовым признаком из df и таргетом.
    Если target=None, то – попарно между числовыми признаками.

    В итоговом датафрейме остаются только числовые признаки.
    """
    df = df.copy()
    if target is not None:
        if isinstance(target, str):
            target_series = df.pop(target)
        elif isinstance(target, pd.Series):
            target_series = target
        else:
            raise ValueError("target должен быть None, str или pd.Series")
        num_features = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]
        results = {}
        for col in num_features:
            results[col] = compute_num_num_metrics(df[col], target_series, nan_policy)
        result_df = pd.DataFrame(results).T
        if p_values_only:
            result_df = result_df[
                [col for col in result_df.columns if col.endswith("_p")]
            ]
        return result_df
    else:
        num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        metric_keys = [
            "pearson",
            "pearson_p",
            "pearson_r2",
            "spearman",
            "spearman_p",
            "spearman_r2",
            "mutual_info",
        ]
        results = {
            key: pd.DataFrame(index=num_cols, columns=num_cols) for key in metric_keys
        }
        for i, col1 in enumerate(num_cols):
            for j in range(i, len(num_cols)):
                col2 = num_cols[j]
                met_dict = compute_num_num_metrics(df[col1], df[col2], nan_policy)
                for key in metric_keys:
                    results[key].at[col1, col2] = met_dict[key]
                    results[key].at[col2, col1] = met_dict[key]
        if p_values_only:
            results = {key: df for key, df in results.items() if key.endswith("_p")}
        return results
