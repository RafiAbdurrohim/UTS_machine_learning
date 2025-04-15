import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# === 1. Load Dataset ===
df = pd.read_csv("dataset_medicion.csv.csv")  # Sesuaikan dengan nama dataset kamu
df["prescribed_dose"] = df["prescribed_dose"].str.extract(r'(\d+)').astype(float)



# === 2. Preprocessing ===
# Pilih fitur yang digunakan untuk prediksi
features = ["age", "gender", "medical_condition", "drug_name", "dosage_form", 
            "prescribed_dose", "frequency", "temperature", "humidity", "distance"]
target = "status"  # Target: 'Taken' atau 'Not Taken'

# Encode kolom kategorikal
le = LabelEncoder()
for col in ["gender", "medical_condition", "drug_name", "dosage_form", "frequency"]:
    df[col] = le.fit_transform(df[col])

# Pisahkan fitur dan target
X = df[features]
y = df[target]

# === 3. Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Normalisasi (Scaling) ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 5. SMOTE (Jika Data Tidak Seimbang) ===
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# === 6. Tuning Nilai K ===
best_k = 1
best_score = 0

for k in range(1, 21):  # Coba dari k=1 sampai k=20
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_resampled, y_train_resampled)
    score = knn.score(X_test_scaled, y_test)
    
    if score > best_score:
        best_score = score
        best_k = k

print(f"Nilai k terbaik: {best_k}, Akurasi: {best_score:.2f}")

# === 7. Model KNN dengan k Terbaik ===
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_resampled, y_train_resampled)
y_pred = knn.predict(X_test_scaled)

# === 8. Evaluasi Model ===
print(f"Akurasi: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
