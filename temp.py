import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('file/train.csv')
test = pd.read_csv('file/test.csv')
test_id = test['id']

print("=" * 60)
print("INITIAL DATA OVERVIEW")
print("=" * 60)
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nClass distribution:\n{train['kelas'].value_counts()}")
print(f"\nMissing values in train:\n{train.isnull().sum()[train.isnull().sum() > 0]}")

# Feature Engineering Function
def engineer_features(df):
    """Create additional features from existing data"""
    df = df.copy()
    
    # Text length features if deskripsi_singkat exists
    if 'deskripsi_singkat' in df.columns:
        df['desc_length'] = df['deskripsi_singkat'].fillna('').astype(str).str.len()
        df['desc_word_count'] = df['deskripsi_singkat'].fillna('').astype(str).str.split().str.len()
    
    # Interaction features for numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) >= 2:
        # Create some interaction features (limit to avoid explosion)
        for i, col1 in enumerate(numerical_cols[:3]):
            for col2 in numerical_cols[i+1:4]:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    return df

# Apply feature engineering
train = engineer_features(train)
test = engineer_features(test)

# Drop unnecessary columns
cols_to_drop = ['id', 'deskripsi_singkat']
train = train.drop([col for col in cols_to_drop if col in train.columns], axis=1)
test = test.drop([col for col in cols_to_drop if col in test.columns], axis=1)

# Prepare target and features
y = train['kelas']
x = train.drop('kelas', axis=1)

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Stratified split with larger validation set for better evaluation
x_train, x_val, y_train, y_val = train_test_split(
    x, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

print(f"\nTrain set: {x_train.shape}, Validation set: {x_val.shape}")

# Identify feature types
numerical_features = x.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = x.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Enhanced preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Median is more robust to outliers
    ('scaler', StandardScaler())  # Scaling helps with feature importance
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=50))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "=" * 60)
print("HYPERPARAMETER OPTIMIZATION")
print("=" * 60)

# Try Optuna for efficient hyperparameter tuning
try:
    import optuna
    from optuna.samplers import TPESampler
    
    print("Using Optuna for hyperparameter optimization...")
    
    def objective(trial):
        # Sample hyperparameters with wider ranges
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=100),
            'max_depth': trial.suggest_int('max_depth', 10, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.7, 0.8]),
            'max_samples': trial.suggest_float('max_samples', 0.7, 1.0),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.01)
        }
        
        clf = RandomForestClassifier(
            random_state=42,
            n_jobs=1,
            class_weight='balanced',
            bootstrap=True,
            **params
        )
        
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', clf)
        ])
        
        scores = cross_val_score(
            pipe, x_train, y_train,
            cv=skf,
            scoring='f1_weighted',
            n_jobs=-1
        )
        return scores.mean()
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=50, show_progress_bar=True, n_jobs=1)
    
    best_params = study.best_params
    print(f"\nBest Optuna parameters: {best_params}")
    print(f"Best CV F1-Score: {study.best_value:.4f}")
    
    # Train final model with best parameters
    final_rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        bootstrap=True,
        **best_params
    )
    
except Exception as e:
    print(f"Optuna optimization failed: {e}")
    print("Using default optimized parameters...")
    
    # Fallback to well-tuned default parameters
    final_rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        max_samples=0.8,
        class_weight='balanced',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

# Create ensemble with multiple models for better robustness
print("\n" + "=" * 60)
print("TRAINING ENSEMBLE MODEL")
print("=" * 60)

# Gradient Boosting as complementary model
gb_clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    min_samples_split=10,
    min_samples_leaf=4,
    subsample=0.8,
    random_state=42
)

# Voting ensemble combines predictions
ensemble = VotingClassifier(
    estimators=[
        ('rf', final_rf),
        ('gb', gb_clf)
    ],
    voting='soft',
    weights=[2, 1]  # Give more weight to Random Forest
)

# Build final pipeline
best_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', ensemble)
])

# Train on full training set
print("Training ensemble model...")
best_pipeline.fit(x_train, y_train)

# Validation evaluation
print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

y_pred_val = best_pipeline.predict(x_val)

accuracy = accuracy_score(y_val, y_pred_val)
f1 = f1_score(y_val, y_pred_val, average='weighted')

print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Validation F1-Score (Weighted): {f1:.4f}")
print(f"\nDetailed Classification Report:")
print(classification_report(y_val, y_pred_val, target_names=le.classes_))

# Retrain on full dataset for final predictions
print("\n" + "=" * 60)
print("FINAL TRAINING ON FULL DATASET")
print("=" * 60)

best_pipeline.fit(x, y_encoded)
print("Training on full dataset completed!")

# Make predictions on test set
print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

final_prediction_encoded = best_pipeline.predict(test)
final_prediction_text = le.inverse_transform(final_prediction_encoded)

# Create submission file
submission = pd.DataFrame({
    'id': test_id,
    'kelas': final_prediction_text
})

print("\nPrediction Distribution:")
print(submission['kelas'].value_counts())
print(f"\nSample predictions:")
print(submission.head(10))

# Save submission
submission_path = 'file/submission_improved.csv'
submission.to_csv(submission_path, index=False)
print(f"\n✓ Submission file saved: {submission_path}")

print("\n" + "=" * 60)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 60)