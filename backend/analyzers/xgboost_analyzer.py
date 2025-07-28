from xgboost import XGBClassifier
from .base_analyzer import BaseAnalyzer

class XGBoostAnalyzer(BaseAnalyzer):
    """XGBoost analyzer with SHAP explainability"""
    
    def create_model(self):
        """Create XGBoost model"""
        return XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0  # Reduce verbosity for cleaner output
        )
    
    def get_model_name(self):
        """Return model name"""
        return "XGBoost"