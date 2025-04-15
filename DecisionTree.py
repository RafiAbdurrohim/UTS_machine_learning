import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# === 1. Load Dataset ===
df = pd.read_csv("dataset_medicion.csv")  # Ganti sesuai nama file dataset
df["prescribed_dose"] = df["prescribed_dose"].str.extract(r'(\d+)').astype(float)

# === 2. Preprocessing ===
features = ["age", "gender", "medical_condition", "drug_name", "dosage_form", 
            "prescribed_dose", "frequency", "temperature", "humidity", "distance"]
target = "status"

# Encode data kategorikal
le = LabelEncoder()
for col in ["gender", "medical_condition", "drug_name", "dosage_form", "frequency"]:
    df[col] = le.fit_transform(df[col])

X = df[features]
y = df[target]

# === 3. Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Normalisasi ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 5. SMOTE untuk data imbalance ===
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# === 6. Buat model Decision Tree ===
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_resampled, y_train_resampled)
y_pred = clf.predict(X_test_scaled)

# === 7. Evaluasi hasil ===
akurasi = accuracy_score(y_test, y_pred)
print(f"\n🔹 Akurasi Model Decision Tree: {akurasi:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🔹 Confusion Matrix:")
print(f"Benar 'Not Taken'  : {cm[0][0]}")
print(f"Salah 'Not Taken'  : {cm[0][1]}")
print(f"Salah 'Taken'      : {cm[1][0]}")
print(f"Benar 'Taken'      : {cm[1][1]}")

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

print("\n🔹 Laporan Klasifikasi (Tabel):")
print(df_report.round(2))
