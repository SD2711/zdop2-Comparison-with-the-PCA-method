# Импортируем библиотеки и стандартные модули, необходимые для всех заданий.
from pathlib import Path
import site
import sys

import matplotlib

matplotlib.use("Agg")  # Сохраняем рисунки без необходимости открывать окно.
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Общие настройки
DATA_FILE = Path("C:/Users/user/Downloads/student+performance/student/student-mat.csv")
OUT_DIR = Path("pca_outputs")
OUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


# Цвета обоих классов используются одинаково на всех графиках.
COLORS = ("#bf4b59", "#187e8a")
CLASS_NAMES = ("Не сдал (G3 < 10)", "Сдал (G3 ≥ 10)")
FEATURES_5 = ["G1", "G2", "studytime", "failures", "absences"]
LOG = []


def log(message=""):
    """Одновременно выводит текст в консоль и записывает в протокол."""
    message = str(message)
    print(message)
    LOG.append(message)


def stats(y_true, y_pred):
    """Возвращает измеримые показатели качества классификации."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def add_covariance_ellipse(ax, mean, covariance, color, label):
    """Рисует эллипс двух стандартных отклонений для заданного класса."""
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0)
    vectors = vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    ellipse = Ellipse(
        xy=mean,
        width=4 * np.sqrt(values[0]),
        height=4 * np.sqrt(values[1]),
        angle=angle,
        fill=False,
        edgecolor=color,
        linewidth=2.2,
        label=label,
        zorder=7,
    )
    ax.add_patch(ellipse)


def plot_decision_comparison(X_train, X_test, y_train, y_test, models):
    """Показывает решения LDA и QDA, ошибки и ковариационные эллипсы."""
    bg = ListedColormap(["#f9dfe0", "#ddf1f0"])
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 5.55), sharex=True, sharey=True)
    x_min = min(X_train[:, 0].min(), X_test[:, 0].min()) - 1.3
    x_max = max(X_train[:, 0].max(), X_test[:, 0].max()) + 1.3
    y_min = min(X_train[:, 1].min(), X_test[:, 1].min()) - 1.3
    y_max = max(X_train[:, 1].max(), X_test[:, 1].max()) + 1.3
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 350),
        np.linspace(y_min, y_max, 350),
    )
    for ax, (name, model) in zip(axes, models.items()):
        grid_predict = model.predict(np.c_[xx.ravel(), yy.ravel()])
        ax.contourf(xx, yy, grid_predict.reshape(xx.shape), cmap=bg, alpha=0.95)
        ax.contour(
            xx,
            yy,
            model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape),
            levels=[0.5],
            colors=["#303b50"],
            linewidths=2,
        )
        # Точки обучающей выборки показывают распределение признаков.
        for cls in (0, 1):
            mask = np.asarray(y_train) == cls
            ax.scatter(
                X_train[mask, 0],
                X_train[mask, 1],
                s=19,
                c=COLORS[cls],
                alpha=0.43,
                edgecolors="none",
                label=CLASS_NAMES[cls],
                zorder=3,
            )
        # Кресты показывают ошибки на отложенной тестовой выборке.
        incorrect = model.predict(X_test) != np.asarray(y_test)
        ax.scatter(
            X_test[incorrect, 0],
            X_test[incorrect, 1],
            marker="x",
            s=75,
            c="#131926",
            linewidths=1.45,
            label="Ошибки на тесте",
            zorder=8,
        )
        means = model.means_
        for cls in (0, 1):
            covariance = model.covariance_ if name == "LDA" else model.covariance_[cls]
            add_covariance_ellipse(
                ax, means[cls], covariance, COLORS[cls], f"2σ: класс {cls}"
            )
            ax.scatter(
                *means[cls],
                marker="*",
                s=155,
                c=COLORS[cls],
                edgecolors="white",
                linewidths=0.85,
                zorder=9,
            )
        acc = accuracy_score(y_test, model.predict(X_test))
        ax.set_title(f"{name}: Accuracy = {acc:.3f}", weight="bold", pad=11)
        ax.set_xlabel("G1 — оценка за первый период")
        ax.grid(alpha=0.12)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    axes[0].set_ylabel("G2 — оценка за второй период")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:5],
        labels[:5],
        loc="lower center",
        ncol=5,
        fontsize=8.7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.025),
    )
    fig.suptitle(
        "LDA и QDA: границы решений и эллипсы ковариации (2σ)",
        fontsize=14,
        weight="bold",
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.945))
    fig.savefig(OUT / "02_lda_qda_boundaries.png", dpi=175, bbox_inches="tight")
    plt.close(fig)


def plot_grades(df):
    """Строит диаграмму оценок и классов исходного датасета."""
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.45))
    grades = df["G3"].value_counts().sort_index().reindex(range(21), fill_value=0)
    axes[0].bar(grades.index, grades.values, width=0.83, color="#375b83")
    axes[0].axvline(9.5, color="#c94b59", ls="--", lw=2, label="Порог сдачи: G3 = 10")
    axes[0].set(
        xlabel="Итоговая оценка G3",
        ylabel="Число учащихся",
        title="Распределение оценок",
    )
    axes[0].set_xticks(range(0, 21, 2))
    axes[0].legend(frameon=False, fontsize=9)
    counts = ((df["G3"] >= 10).astype(int)).value_counts().reindex([0, 1])
    bars = axes[1].bar(CLASS_NAMES, counts.values, color=COLORS, width=0.53)
    for bar in bars:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4,
            str(round(bar.get_height())),
            ha="center",
            weight="bold",
        )
    axes[1].set(title="Баланс классов", ylabel="Число учащихся")
    axes[1].set_ylim(0, 300)
    fig.tight_layout()
    fig.savefig(OUT / "01_data_overview.png", dpi=175, bbox_inches="tight")
    plt.close(fig)


def plot_confusions(y_test, predictions):
    """Рисует матрицы ошибок двух классификаторов для пяти признаков."""
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.4))
    for ax, (name, predicted) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, predicted, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Не сдал", "Сдал"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{name}: тестовая выборка", weight="bold")
        ax.set_xlabel("Прогноз")
        ax.set_ylabel("Фактический класс")
    fig.tight_layout()
    fig.savefig(OUT / "03_confusion_matrices.png", dpi=175, bbox_inches="tight")
    plt.close(fig)


def plot_lda_pca(X_scaled_train, X_scaled_test, y_train, y_test, pca, lda_project):
    """Сопоставляет двухмерную проекцию PCA и одномерную проекцию LDA."""
    pca_test = pca.transform(X_scaled_test)
    lda_test = lda_project.transform(X_scaled_test)[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 4.7))
    for cls in [0, 1]:
        mask = np.asarray(y_test) == cls
        axes[0].scatter(
            pca_test[mask, 0],
            pca_test[mask, 1],
            c=COLORS[cls],
            s=42,
            alpha=0.75,
            label=CLASS_NAMES[cls],
            edgecolor="white",
            linewidth=0.3,
        )
        rng = np.random.default_rng(2026 + cls)
        jitter = rng.uniform(-0.14, 0.14, size=mask.sum())
        axes[1].scatter(
            lda_test[mask],
            jitter,
            c=COLORS[cls],
            s=42,
            alpha=0.76,
            label=CLASS_NAMES[cls],
            edgecolor="white",
            linewidth=0.3,
        )
    axes[0].set(
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% дисперсии)",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% дисперсии)",
        title="PCA: 2 главные компоненты",
    )
    axes[1].set(
        xlabel="LD1 — дискриминантная компонента", title="LDA: 1 дискриминантная ось"
    )
    axes[1].set_yticks([])
    axes[1].axhline(0, color="gray", lw=0.5)
    for ax in axes:
        ax.grid(alpha=0.16)
    fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "Одни и те же тестовые объекты в пространствах PCA и LDA", weight="bold", y=1
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.94))
    fig.savefig(OUT / "04_pca_vs_lda_projection.png", dpi=175, bbox_inches="tight")
    plt.close(fig)


def plot_pca_variance(pca_full):
    """Отображает долю объяснённой дисперсии всех пяти компонент PCA."""
    ratios = 100 * pca_full.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    xs = np.arange(1, len(ratios) + 1)
    bars = ax.bar(xs, ratios, color="#577899", alpha=0.82, label="Отдельная компонента")
    ax.plot(xs, cumulative, "o-", color="#cc6e43", lw=2, label="Накопленная доля")
    for x, y in zip(xs, cumulative):
        ax.annotate(
            f"{y:.1f}%",
            (x, y),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8.7,
        )
    ax.set(
        xlabel="Номер главной компоненты",
        ylabel="Объяснённая дисперсия, %",
        title="PCA: объяснённая дисперсия исходных пяти признаков",
        ylim=(0, 109),
    )
    ax.set_xticks(xs)
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "05_pca_variance.png", dpi=175, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(summary):
    """Сравнивает качество базовых моделей и моделей после PCA."""
    labels = ["LDA (5 признаков)", "QDA (5 признаков)", "PCA(2) + LDA", "PCA(2) + QDA"]
    accuracy = [summary[k]["Accuracy"] for k in labels]
    f1 = [summary[k]["F1"] for k in labels]
    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.7, 4.65))
    ax.bar(xs - 0.17, accuracy, width=0.34, color="#286b87", label="Accuracy")
    ax.bar(xs + 0.17, f1, width=0.34, color="#cf8754", label="F1")
    for i, (acc, f) in enumerate(zip(accuracy, f1)):
        ax.text(i - 0.17, acc + 0.015, f"{acc:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.17, f + 0.015, f"{f:.3f}", ha="center", fontsize=9)
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, 1.04)
    ax.set_title("Качество классификации на одной тестовой выборке", weight="bold")
    ax.set_ylabel("Значение метрики")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "06_metric_comparison.png", dpi=175, bbox_inches="tight")
    plt.close(fig)


# Задание 1. Выводим реальные пути текущей среды Python и библиотек.
# Путь к PyCharm и снимки Проводника нужно получить на компьютере студента.
log("ЗАДАНИЕ 1. ПУТИ УСТАНОВКИ")
log(f"Исполняемый Python: {sys.executable}")
log(f"Каталог окружения: {sys.prefix}")
log(f"Внешние библиотеки: {site.getsitepackages()}")
log(
    "PyCharm: путь определяется на компьютере пользователя; требуется скриншот Проводника."
)
log()

# Задание 2. Загружаем исходные данные UCI через разделитель ';'.
df = pd.read_csv(CSV, sep=";")
required = ["G1", "G2", "G3", "studytime", "failures", "absences"]
if not set(required).issubset(df.columns):
    raise ValueError(
        "В CSV отсутствуют необходимые столбцы: " + str(set(required) - set(df.columns))
    )
if df[required].isna().any().any():
    raise ValueError("В используемых признаках или в целевой переменной есть пропуски.")

# Формируем двоичную целевую переменную: 0 — не сдал, 1 — сдал.
y = (df["G3"] >= 10).astype(int).to_numpy()
log("ЗАДАНИЕ 2. ОПИСАНИЕ ДАННЫХ")
log(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
log(f"Пропусков во всех столбцах: {int(df.isna().sum().sum())}")
log(f"Уникальных строк: {len(df.drop_duplicates())}")
log(f"Класс 0 (G3 < 10): {(y == 0).sum()} объектов")
log(f"Класс 1 (G3 >= 10): {(y == 1).sum()} объектов")
log("Первые три строки выбранных числовых признаков:")
log(df[required].head(3).to_string(index=False))
plot_grades(df)
log()

# Задание 3. Разделяем объекты на обучающую (75%) и тестовую (25%) части.
# Во всех экспериментах ниже используем одинаковые индексы и random_state=42.
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices, test_size=0.25, random_state=42, stratify=y
)
y_train, y_test = y[train_idx], y[test_idx]
X2 = df[["G1", "G2"]].to_numpy(dtype=float)
X2_train, X2_test = X2[train_idx], X2[test_idx]

# Строим двухмерные модели для визуализации границ и эллипсов ковариации.
lda_2d = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
qda_2d = QuadraticDiscriminantAnalysis(store_covariance=True)
lda_2d.fit(X2_train, y_train)
qda_2d.fit(X2_train, y_train)
pred_lda2 = lda_2d.predict(X2_test)
pred_qda2 = qda_2d.predict(X2_test)
plot_decision_comparison(
    X2_train, X2_test, y_train, y_test, {"LDA": lda_2d, "QDA": qda_2d}
)

# Для более полного сравнения используем пять признаков с нормированием.
X5 = df[FEATURES_5].to_numpy(dtype=float)
scaler = StandardScaler()
X_train = scaler.fit_transform(X5[train_idx])
X_test = scaler.transform(X5[test_idx])
lda_5d = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
qda_5d = QuadraticDiscriminantAnalysis(store_covariance=True)
lda_5d.fit(X_train, y_train)
qda_5d.fit(X_train, y_train)
pred_lda5 = lda_5d.predict(X_test)
pred_qda5 = qda_5d.predict(X_test)
plot_confusions(y_test, {"LDA": pred_lda5, "QDA": pred_qda5})

log("ЗАДАНИЕ 3. LDA И QDA")
log(f"Обучающая выборка: {len(train_idx)}; тестовая выборка: {len(test_idx)}")
summary = {}
for name, predictions in [
    ("LDA (2 признака)", pred_lda2),
    ("QDA (2 признака)", pred_qda2),
    ("LDA (5 признаков)", pred_lda5),
    ("QDA (5 признаков)", pred_qda5),
]:
    summary[name] = stats(y_test, predictions)
    log(
        f"{name}: Accuracy={summary[name]['Accuracy']:.4f}, "
        f"BalancedAcc={summary[name]['Balanced accuracy']:.4f}, "
        f"F1={summary[name]['F1']:.4f}"
    )
    log(f"Матрица ошибок: {confusion_matrix(y_test, predictions).tolist()}")
log()

# Задание 4. Обучаем PCA только на обучающей части пяти признаков.
# PCA не является классификатором: оцениваем PCA(2) + LDA / PCA(2) + QDA.
pca_all = PCA(n_components=len(FEATURES_5))
pca_all.fit(X_train)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
lda_pca = LinearDiscriminantAnalysis(solver="svd")
qda_pca = QuadraticDiscriminantAnalysis()
lda_pca.fit(X_train_pca, y_train)
qda_pca.fit(X_train_pca, y_train)
pred_lda_pca = lda_pca.predict(X_test_pca)
pred_qda_pca = qda_pca.predict(X_test_pca)

# LDA-проекция для двух классов может иметь только одну ось (n_classes - 1).
lda_proj = LinearDiscriminantAnalysis(n_components=1)
lda_proj.fit(X_train, y_train)
plot_lda_pca(X_train, X_test, y_train, y_test, pca, lda_proj)
plot_pca_variance(pca_all)

summary["PCA(2) + LDA"] = stats(y_test, pred_lda_pca)
summary["PCA(2) + QDA"] = stats(y_test, pred_qda_pca)
plot_metric_comparison(summary)
log("ЗАДАНИЕ 4. СРАВНЕНИЕ С PCA")
log(
    "Доли дисперсии по пяти компонентам: "
    + ", ".join(f"{val:.4f}" for val in pca_all.explained_variance_ratio_)
)
log(
    f"Суммарная доля для первых двух компонент: {pca.explained_variance_ratio_.sum():.4f}"
)
for name in ["LDA (5 признаков)", "QDA (5 признаков)", "PCA(2) + LDA", "PCA(2) + QDA"]:
    scores = summary[name]
    log(
        f"{name}: Accuracy={scores['Accuracy']:.4f}; "
        f"BalancedAcc={scores['Balanced accuracy']:.4f}; "
        f"Precision={scores['Precision']:.4f}; "
        f"Recall={scores['Recall']:.4f}; F1={scores['F1']:.4f}"
    )
log()

# Задания 5–6. Отчёт оформляется в Word, а код, CSV и отчёт передаются в ЭОР.
log("ЗАДАНИЯ 5–6. ПОДГОТОВКА К СДАЧЕ")
log("К заданию прикладываются zdop2_Benallal.docx, zdop2_Benallal.py и student-mat.csv")
log("Нужно добавить в Word реальные скриншоты путей Python, PyCharm и библиотек.")
log("Готово: графики и протокол сохранены в папку zdop2_results.")
(OUT / "console_output.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")

# Сохраняем таблицу вычисленных метрик, чтобы сверить числа с отчётом.
(
    pd.DataFrame(summary)
    .T.round(6)
    .rename_axis("Метод")
    .to_csv(OUT / "metrics.csv", encoding="utf-8-sig")
)
