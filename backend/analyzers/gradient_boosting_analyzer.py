from sklearn.ensemble import GradientBoostingClassifier
from .base_analyzer import BaseAnalyzer
import numpy as np
from sklearn.preprocessing import LabelEncoder

class GradientBoostingAnalyzer(BaseAnalyzer):
    """Gradient Boosting analyzer with SHAP explainability and robust error handling"""

    def create_model(self):
        """Create Gradient Boosting model with robust parameters"""
        return GradientBoostingClassifier(
            n_estimators=100,  # Reduced from 200 to prevent memory issues
            learning_rate=0.1,
            max_depth=4,       # Reduced from 6 to prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
            verbose=0,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4
        )

    def preprocess_data(self, X, y=None):
        """Clean input data and encode labels"""
        # Check for infinite values
        if np.any(np.isinf(X)):
            print("[GB] Warning: Infinite values found in features, clipping")
            X = np.clip(X, -1e10, 1e10)

        # Replace NaN and infinities with finite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

        if y is not None:
            if len(np.unique(y)) < 2:
                raise ValueError("Need at least 2 classes for classification")

            if not np.issubdtype(y.dtype, np.integer):
                le = LabelEncoder()
                y = le.fit_transform(y)

        return X, y

    def fit_model_with_validation(self, model, X_train, y_train, X_test, y_test):
        """Fit model with enhanced validation and error handling"""
        try:
            print(f"[GB] Training data shape: {X_train.shape}")
            print(f"[GB] Training labels shape: {y_train.shape}")
            print(f"[GB] Unique classes: {np.unique(y_train)}")

            X_train_processed, y_train_processed = self.preprocess_data(X_train, y_train)
            X_test_processed, _ = self.preprocess_data(X_test)

            if len(X_train_processed) < 10:
                raise ValueError("Insufficient training data: Need at least 10 samples")

            print("[GB] Starting model training...")
            model.fit(X_train_processed, y_train_processed)

            if not hasattr(model, 'estimators_') or model.estimators_ is None:
                raise RuntimeError("Model training failed: estimators_ is None")

            if len(model.estimators_) == 0:
                raise RuntimeError("Model training failed: no estimators created")

            none_estimators = sum(1 for stage in model.estimators_ for est in stage if est is None)
            if none_estimators > 0:
                raise RuntimeError(f"Model training partially failed: {none_estimators} None estimators")

            print(f"[GB] Model trained successfully with {len(model.estimators_)} estimators")

            # Test prediction on a sample
            test_sample = X_test_processed[:min(5, len(X_test_processed))]
            try:
                test_pred = model.predict(test_sample)
                print(f"[GB] Test prediction successful: {test_pred}")
            except Exception as pred_error:
                raise RuntimeError(f"Model prediction test failed: {pred_error}")

            return model, X_train_processed, y_train_processed, X_test_processed, y_test

        except Exception as e:
            print(f"[GB] Model training failed: {e}")
            print("[GB] Attempting fallback with conservative parameters...")
            fallback_model = GradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.05,
                max_depth=3,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.7,
                random_state=42,
                verbose=0
            )
            try:
                X_train_processed, y_train_processed = self.preprocess_data(X_train, y_train)
                X_test_processed, _ = self.preprocess_data(X_test)

                fallback_model.fit(X_train_processed, y_train_processed)

                if not hasattr(fallback_model, 'estimators_') or fallback_model.estimators_ is None:
                    raise RuntimeError("Fallback model failed: estimators_ is None")

                if len(fallback_model.estimators_) == 0:
                    raise RuntimeError("Fallback model failed: no estimators")

                print("[GB] Fallback model trained successfully")
                return fallback_model, X_train_processed, y_train_processed, X_test_processed, y_test

            except Exception as fallback_error:
                raise RuntimeError(f"Both primary and fallback model training failed.\nPrimary: {e}\nFallback: {fallback_error}")

    def get_model_name(self):
        """Return model name"""
        return "Gradient Boosting"
