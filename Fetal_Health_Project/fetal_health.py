import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
from collections import Counter

# Tüm sütunların eksiksiz görüntülenmesini sağlamadım
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

data = pd.read_csv("fetal_health.csv")

# Eksik veri kontrolü ve doldurma
missing_before = data.isnull().sum()
data.fillna(data.mean(), inplace=True)
missing_after = data.isnull().sum()
missing_comparison = pd.DataFrame({
    "Eksik Öncesi": missing_before,
    "Eksik Sonrası": missing_after
})
print("\nEksik Değerler Doldurulmadan Önce ve Sonra:")
print(missing_comparison)

# Korelasyon Matrisi
plt.figure(figsize=(14, 14))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            cbar=True, square=True, linewidths=.5, annot_kws={"size":8})
plt.title("Korelasyon Matrisi", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

#Feature Engineering
data["total_decelerations"] = (
    data["light_decelerations"] +
    data["severe_decelerations"] +
    data["prolongued_decelerations"]
)
data["variability_ratio"] = (
    data["mean_value_of_short_term_variability"] /
    data["mean_value_of_long_term_variability"]
).replace([np.inf, -np.inf], np.nan).fillna(0)

# Hedef değişkeni yeniden kodlama
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data["fetal_health"])
y = y_encoded
X = data.drop(columns=["fetal_health"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Özellik önem düzeyi analizi ve görselleştirilmesi(Random Forest)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_scaled, y)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances_sorted.values, y=feature_importances_sorted.index)
plt.title("Özellik Önem Düzeyi (Random Forest)", fontsize=16)
plt.xlabel("Önem Düzeyi", fontsize=12)
plt.ylabel("Özellikler", fontsize=12)
plt.tight_layout()
plt.show()

# En az önemli özelliklerin çıkarılması (eşik değeri belirleyerek)
importance_threshold = 0.01
important_features = feature_importances[feature_importances > importance_threshold].index
print(f"Seçilen önemli özellikler ({len(important_features)} özellik):")
print(important_features)

# Önemli özelliklerle yeni veri seti oluşturma
X_important = X[important_features]

# Veriyi eğitim ve test setlerine ayırma (Ölçeklendirmeden önce)
X_train, X_test, y_train, y_test = train_test_split(
    X_important, y, test_size=0.2, random_state=42, stratify=y
)

# Özellik ölçeklendirme (sadece eğitim verisi üzerinde fit edilir)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE (eğitim setine)
smote = SMOTE(random_state=43)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print("SMOTE öncesi sınıf dağılımı:", Counter(y_train))
print("SMOTE sonrası sınıf dağılımı:", Counter(y_train_smote))

# Modeller:
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42, C=0.1
    ),
    "Random Forest": RandomForestClassifier(
        random_state=42, n_estimators=100, max_depth=5, min_samples_leaf=10
    ),
    "Support Vector Machine": SVC(
        probability=True, kernel='rbf', C=0.5, random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42, max_depth=5, min_samples_leaf=10
    ),
    "XGBoost": XGBClassifier(
        random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1, reg_lambda=10
    )
}

# Model değerlendirme metriklerini saklama
metrics = []
for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_test_pred = model.predict(X_test_scaled)

    # Metrikleri hesaplama
    metrics.append({
        "Model": name,
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred, average="weighted"),
        "Recall": recall_score(y_test, y_test_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_test_pred, average="weighted")
    })

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[str(i) for i in label_encoder.classes_],
                yticklabels=[str(i) for i in label_encoder.classes_])
    plt.title(f"Confusion Matrix: {name}", fontsize=14)
    plt.xlabel("Tahmin Edilen", fontsize=12)
    plt.ylabel("Gerçek", fontsize=12)
    plt.tight_layout()
    plt.show()

    # Sınıflandırma Raporu
    target_names = [str(i) for i in label_encoder.classes_]
    print(f"Sınıflandırma Raporu - {name}:")
    print(classification_report(y_test, y_test_pred, target_names=target_names))

    # Her sınıf için ROC Eğrisi
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_binarized.shape[1]
    y_score = model.predict_proba(X_test_scaled)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(n_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='Class {0} ROC (AUC = {1:0.2f})'
                 ''.format(label_encoder.classes_[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f"ROC Curves - {name}", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.show()

# Performans sonuçları
metrics_df = pd.DataFrame(metrics)
print(metrics_df)

# Karşılaştırma grafiği
metrics_df.set_index("Model")[["Test Accuracy", "F1 Score"]].plot(
    kind="bar", figsize=(10, 6), ylim=(0.7, 1.0)
)
plt.title("Performance of Different Classifier Models", fontsize=16)
plt.ylabel("Score", fontsize=12)
plt.xticks(rotation=45)
plt.legend(loc="lower right", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
