import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, accuracy_score, classification_report

train = pd.read_csv('file/train.csv')
test = pd.read_csv('file/test.csv')
test_id = test['id']

print("train: ", train.shape)
print("test: ", test.shape)

print("Train info: ")
print(train.info())
print(train['kelas'].value_counts())
train.head()

train = train.drop(['id', 'deskripsi_singkat'], axis=1)
test = test.drop(['id', 'deskripsi_singkat'], axis=1)

y = train['kelas']
x = train.drop('kelas', axis=1)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

x_train, x_val, y_train, y_val = train_test_split(
    x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

numerical_features = x.select_dtypes(include=['int64', 'float64']).columns
categorical_features = x.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features), 
        ('cat', categorical_transformer, categorical_features)
    ]
)

main_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('model', RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'))
])

# Grid search yang lebih ekstensif
param_grid = {
    'model__n_estimators': [200, 300, 400],
    'model__max_depth': [15, 20, None],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2],
    'model__max_features': ['sqrt', 'log2']
}

scorer = make_scorer(f1_score, average='weighted')

# Menggunakan StratifiedKFold untuk cross-validation yang lebih baik
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    main_pipeline, 
    param_grid, 
    scoring=scorer, 
    cv=skf, 
    verbose=2, 
    n_jobs=-1
)

grid_search.fit(x_train, y_train)

print(f"\nSetelan terbaik: {grid_search.best_params_}")
print(f"Skor F1 terbaik (CV): {grid_search.best_score_:.4f}")

best_pipeline = grid_search.best_estimator_

y_pred_val_tuned = best_pipeline.predict(x_val)

akurasi_tuned = accuracy_score(y_val, y_pred_val_tuned)
f1_tuned = f1_score(y_val, y_pred_val_tuned, average='weighted')

print(f"\nAkurasi model: {akurasi_tuned * 100:.2f}%")
print(f"F1-Score (Tuned, Weighted): {f1_tuned:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_val_tuned, target_names=le.classes_))

# Feature importance analysis
if hasattr(best_pipeline.named_steps['model'], 'feature_importances_'):
    print("\nAnalyzing feature importance...")

final_prediction_encoded = best_pipeline.predict(test)
final_prediction_text = le.inverse_transform(final_prediction_encoded)

laporan_akhir = pd.DataFrame({
    'id': test_id,
    'kelas': final_prediction_text
})

print("\nHasil akhir:")
print(laporan_akhir.head())
print(f"\nDistribusi prediksi:")
print(laporan_akhir['kelas'].value_counts())

submission_path = 'file/submissiontest5.csv'
laporan_akhir.to_csv(submission_path, index=False)
print(f"\nFile submission berhasil disimpan: {submission_path}")