from sklearn.ensemble import RandomForestClassifier
from .base_analyzer import BaseAnalyzer

class RandomForestAnalyzer(BaseAnalyzer):
    """Random Forest analyzer with SHAP explainability"""
    
    def create_model(self):
        """Create Random Forest model"""
        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    
    def get_model_name(self):
        """Return model name"""
        return "Random Forest"