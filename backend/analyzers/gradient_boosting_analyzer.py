from sklearn.ensemble import GradientBoostingClassifier
from .base_analyzer import BaseAnalyzer

class GradientBoostingAnalyzer(BaseAnalyzer):
    """Gradient Boosting analyzer with SHAP explainability"""
    
    def create_model(self):
        """Create Gradient Boosting model"""
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=0  # Reduce verbosity for cleaner output
        )
    
    def get_model_name(self):
        """Return model name"""
        return "Gradient Boosting"