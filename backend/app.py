from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from analyzers.random_forest_analyzer import RandomForestAnalyzer
from analyzers.xgboost_analyzer import XGBoostAnalyzer
from analyzers.gradient_boosting_analyzer import GradientBoostingAnalyzer

app = Flask(__name__)
CORS(app)

# Initialize analyzersa
analyzers = {
    'random_forest': RandomForestAnalyzer(),
    'xgboost': XGBoostAnalyzer(),
    'gradient_boosting': GradientBoostingAnalyzer()
}

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    try:
        print("[API] /api/analyze endpoint hit.")
        print("🚀 Started at:", datetime.now())
        
        cipher_file = request.files.get('cipher_file')
        random_file = request.files.get('random_file')
        model_type = request.form.get('model_type', 'random_forest')
        
        if not cipher_file or not random_file:
            print("[API] Error: Both files are required.")
            return jsonify({'error': 'Both files are required'}), 400
        
        if model_type not in analyzers:
            print(f"[API] Error: Invalid model type: {model_type}")
            return jsonify({'error': f'Invalid model type. Available: {list(analyzers.keys())}'}), 400
        
        print(f"[API] Using model: {model_type}")
        analyzer = analyzers[model_type]
        
        # Run the complete analysis pipeline
        results = analyzer.run_complete_analysis(cipher_file, random_file)
        
        print("[API] Analysis complete. Returning results.")
        return jsonify(results)
        
    except Exception as e:
        print(f"[API] Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    print("[API] /api/health endpoint hit.")
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'available_models': list(analyzers.keys())
    })

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get information about available models"""
    models_info = {
        'random_forest': {
            'name': 'Random Forest',
            'description': 'Ensemble method using multiple decision trees',
            'strengths': ['Fast training', 'Good interpretability', 'Handles overfitting well']
        },
        'xgboost': {
            'name': 'XGBoost',
            'description': 'Gradient boosting framework optimized for speed and performance',
            'strengths': ['High accuracy', 'Fast SHAP computation', 'Advanced regularization']
        },
        'gradient_boosting': {
            'name': 'Gradient Boosting',
            'description': 'Sequential ensemble method building models iteratively',
            'strengths': ['Good performance', 'Built-in feature selection', 'Robust to outliers']
        }
    }
    return jsonify(models_info)

if __name__ == '__main__':
    try:
        print("[MAIN] Starting Flask server with multiple ML models...")
        print(f"[MAIN] Available models: {list(analyzers.keys())}")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"[MAIN] Failed to start server: {e}")
        import traceback
        traceback.print_exc()