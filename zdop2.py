from pathlib import Path
import sys
import site

import matplotlib

matplotlib.use("Agg")  # Сохраняем графики в PNG (подходит для VS Code).
import matplotlib.pyplot as plt
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
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ВАЖНО: student-mat.csv должен лежать в одной папке с zdop2.py!
PAPKA = Path(__file__).resolve().parent
DATASET = PAPKA / "student-mat.csv"
RESULTS = PAPKA / "zdop2_results"
RESULTS.mkdir(exist_ok=True)

if not DATASET.exists():
    raise FileNotFoundError(
        f"Не найден {DATASET}\n"
        "C:\\Users\\user\\OneDrive\\Desktop\\Git repository\\zdop2-Comparison-with-the-PCA-method\\student-mat.csv."
    )

# Задание 1. Показываем пути Python и библиотек.
print("ЗАДАНИЕ 1: ПУТИ УСТАНОВКИ")
print("Python:", sys.executable)
print("Библиотеки:", site.getsitepackages())
print("Для PyCharm нужен скриншот пути установки на вашем ПК.\n")

# Задание 2. Читаем CSV (у student-mat разделитель — точка с запятой).
df = pd.read_csv(DATASET, sep=";")
features = ["G1", "G2", "studytime", "failures", "absences"]
needed = features + ["G3"]
if any(c not in df.columns for c in needed):
    raise ValueError("В CSV нет столбцов: " + str(set(needed) - set(df.columns)))
if df[needed].isna().any().any():
    raise ValueError("В числовых признаках есть пропуски.")
y = (df["G3"] >= 10).astype(int).to_numpy()  # 1 — сдал, 0 — не сдал

print("ЗАДАНИЕ 2: ОПИСАНИЕ ДАННЫХ")
print(
    "Источник: C:\\Users\\user\\OneDrive\\Desktop\\Git repository\\zdop2-Comparison-with-the-PCA-method\\student-mat.csv"
)
print("Размер:", df.shape, "| Не сдал:", (y == 0).sum(), "| Сдал:", (y == 1).sum())
print("Пропуски:", df.isna().sum().sum())
print(df[needed].head(), "\n")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(df["G3"], bins=np.arange(-0.5, 21.5, 1), edgecolor="white")
ax[0].axvline(9.5, ls="--", color="red", label="Порог 10")
ax[0].set(title="Итоговые оценки G3", xlabel="Оценка", ylabel="Число учеников")
ax[0].legend()
ax[1].bar(["Не сдал", "Сдал"], [(y == 0).sum(), (y == 1).sum()])
ax[1].set(title="Количество учеников по классам", ylabel="Количество")
fig.tight_layout()
fig.savefig(RESULTS / "01_data_overview.png", dpi=160)
plt.close(fig)

# Все модели тестируем на одних и тех же учениках.
train, test = train_test_split(
    np.arange(len(y)), test_size=0.25, stratify=y, random_state=42
)
y_train, y_test = y[train], y[test]

# Задание 3. Обучаем LDA и QDA по двум оценкам G1 и G2.
X2 = df[["G1", "G2"]].to_numpy(dtype=float)
lda2 = LinearDiscriminantAnalysis(store_covariance=True).fit(X2[train], y_train)
qda2 = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X2[train], y_train)

print("ЗАДАНИЕ 3: LDA и QDA")
for name, model in [("LDA (2)", lda2), ("QDA (2)", qda2)]:
    pred = model.predict(X2[test])
    print(
        f"{name}: Accuracy={accuracy_score(y_test, pred):.4f}, F1={f1_score(y_test, pred):.4f}"
    )

# Рисуем области классификации и эллипсы ковариации (2 стандартных отклонения).
xx, yy = np.meshgrid(np.linspace(-1, 21, 250), np.linspace(-1, 21, 250))
grid = np.column_stack([xx.ravel(), yy.ravel()])
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, model) in zip(axes, [("LDA", lda2), ("QDA", qda2)]):
    Z = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#ffe3e5", "#dff4f1"])
    for cls, color in [(0, "#b64252"), (1, "#11818b")]:
        points = X2[train][y_train == cls]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=18,
            alpha=0.6,
            color=color,
            label="Не сдал" if cls == 0 else "Сдал",
        )
        cov = model.covariance_ if name == "LDA" else model.covariance_[cls]
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        v = eigenvalues[order].clip(min=0)
        direction = eigenvectors[:, order[0]]
        angle = np.degrees(np.arctan2(direction[1], direction[0]))
        ax.add_patch(
            Ellipse(
                model.means_[cls],
                4 * np.sqrt(v[0]),
                4 * np.sqrt(v[1]),
                angle=angle,
                fill=False,
                edgecolor=color,
                lw=2,
            )
        )
    errors = model.predict(X2[test]) != y_test
    ax.scatter(
        X2[test][errors, 0],
        X2[test][errors, 1],
        c="black",
        marker="x",
        s=55,
        label="Ошибки",
    )
    ax.set(xlabel="G1", ylabel="G2", title=f"{name}: границы и эллипсы")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RESULTS / "02_lda_qda_boundaries.png", dpi=160)
plt.close(fig)

# Обучаем те же классификаторы на пяти стандартизированных признаках.
X5 = df[features].to_numpy(dtype=float)
scaler = StandardScaler()
Xtrain = scaler.fit_transform(X5[train])
Xtest = scaler.transform(X5[test])
lda5 = LinearDiscriminantAnalysis().fit(Xtrain, y_train)
qda5 = QuadraticDiscriminantAnalysis().fit(Xtrain, y_train)

# Задание 4. PCA сокращает пять признаков до двух компонент.
pca = PCA(n_components=2)
P_train = pca.fit_transform(Xtrain)
P_test = pca.transform(Xtest)
lda_pca = LinearDiscriminantAnalysis().fit(P_train, y_train)
qda_pca = QuadraticDiscriminantAnalysis().fit(P_train, y_train)

models = {
    "LDA (5 признаков)": (lda5, Xtest),
    "QDA (5 признаков)": (qda5, Xtest),
    "PCA(2) + LDA": (lda_pca, P_test),
    "PCA(2) + QDA": (qda_pca, P_test),
}
results = {}
print("\nЗАДАНИЕ 4: СРАВНЕНИЕ С PCA")
print(
    "Доля дисперсии первых двух компонент PCA:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%",
)
for name, (model, data) in models.items():
    prediction = model.predict(data)
    acc = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    results[name] = {"Accuracy": acc, "F1": f1}
    print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")

# Матрицы ошибок для LDA и QDA (5 признаков).
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, model) in zip(axes, [("LDA", lda5), ("QDA", qda5)]):
    cm = confusion_matrix(y_test, model.predict(Xtest))
    ConfusionMatrixDisplay(cm, display_labels=["Не сдал", "Сдал"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"{name}: матрица ошибок")
fig.tight_layout()
fig.savefig(RESULTS / "03_confusion_matrices.png", dpi=160)
plt.close(fig)

# Сравнение проекций: PCA даёт 2 оси, LDA для двух классов — одну.
lda_axis = LinearDiscriminantAnalysis(n_components=1).fit(Xtrain, y_train)
L_test = lda_axis.transform(Xtest).ravel()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for cls, color, name in [(0, "#b64252", "Не сдал"), (1, "#11818b", "Сдал")]:
    mask = y_test == cls
    axes[0].scatter(P_test[mask, 0], P_test[mask, 1], c=color, s=30, label=name)
    axes[1].scatter(
        L_test[mask], np.zeros(mask.sum()) + cls * 0.12, c=color, s=30, label=name
    )
axes[0].set(title="PCA: 2 компоненты", xlabel="PC1", ylabel="PC2")
axes[1].set(title="LDA: 1 компонента", xlabel="LD1", yticks=[])
axes[0].legend()
fig.tight_layout()
fig.savefig(RESULTS / "04_pca_vs_lda_projection.png", dpi=160)
plt.close(fig)

# Дисперсия PCA по всем пяти главным компонентам.
all_pca = PCA().fit(Xtrain)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, 6), all_pca.explained_variance_ratio_ * 100)
ax.set(
    title="PCA: доля дисперсии каждой компоненты",
    xlabel="Главная компонента",
    ylabel="Дисперсия (%)",
)
fig.tight_layout()
fig.savefig(RESULTS / "05_pca_variance.png", dpi=160)
plt.close(fig)

# Общая диаграмма точности методов.
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(list(results), [item["Accuracy"] for item in results.values()])
ax.set(title="Точность LDA, QDA и PCA + классификация", ylabel="Accuracy", ylim=(0, 1))
ax.tick_params(axis="x", labelrotation=10)
fig.tight_layout()
fig.savefig(RESULTS / "06_metric_comparison.png", dpi=160)
plt.close(fig)

# Сохраняем числовые результаты.
pd.DataFrame(results).T.to_csv(RESULTS / "metrics.csv", encoding="utf-8-sig")
print("\nГотово! Все графики и metrics.csv находятся в папке:")
print(RESULTS)
