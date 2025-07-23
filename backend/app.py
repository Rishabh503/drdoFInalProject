from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import warnings
import io
import base64
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

plt.switch_backend('Agg')
sns.set_style("whitegrid")

class ShapAnalyzer:
    def __init__(self):
        print("[INIT] ShapAnalyzer initialized.")
        self.model = None
        self.explainer = None
        self.X_test = None
        self.y_test = None
        self.shap_values = None
        self.shap_values_pos = None
        self.feature_names = None
        self.X = None
        self.y = None
        
    def process_data(self, cipher_file, random_file):
        print("[process_data] Reading cipher and random files.")
        cipher_df = pd.read_csv(cipher_file, header=None)
        random_df = pd.read_csv(random_file, header=None)
        
        if cipher_df.shape[1] != random_df.shape[1]:
            print("[process_data] Error: Shape mismatch!")
            raise Exception(f"Shape mismatch: Cipher has {cipher_df.shape[1]} columns, Random has {random_df.shape[1]} columns")
        
        print(f"[process_data] Cipher shape: {cipher_df.shape}, Random shape: {random_df.shape}")

        cipher_df['label'] = 1
        random_df['label'] = 0
        
        combined_df = pd.concat([cipher_df, random_df], ignore_index=True)
        combined_df.dropna(inplace=True)
        
        X = combined_df.drop(columns=['label'])
        y = combined_df['label']
        
        # Rename columns to bit positions
        self.feature_names = [f'bit_{i}' for i in range(X.shape[1])]
        X.columns = self.feature_names
        
        self.X = X.copy()
        self.y = y.copy()
        
        # Split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.X_test = X_test
        self.y_test = y_test

        print("[process_data] Data processing complete. Returning train/test splits.")
        return X_train, X_test, y_train, y_test, X, y
    
    def train_model(self, X_train, y_train):
        print("[train_model] Training RandomForestClassifier.")
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        print("[train_model] Model trained.")
        return self.model
    
    def generate_shap_values(self):
        print("[generate_shap_values] Creating SHAP explainer and computing SHAP values.")
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.X_test)
        
        # Handle both binary and multi-class cases
        if isinstance(self.shap_values, list):
            if len(self.shap_values) == 2:
                # Binary classification - use positive class SHAP values
                self.shap_values_pos = self.shap_values[1]
            else:
                # Multi-class - use first class for now
                self.shap_values_pos = self.shap_values[0]
        else:
            # Single array returned
            self.shap_values_pos = self.shap_values
        
        # Ensure we have a 2D array
        if self.shap_values_pos.ndim == 3:
            self.shap_values_pos = self.shap_values_pos[:, :, 0]
                
        print(f"[generate_shap_values] SHAP values generated. Shape: {self.shap_values_pos.shape}")
        return self.shap_values_pos
    
    def create_summary_plot(self):
        print("[create_summary_plot] Creating SHAP summary plot.")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values_pos, self.X_test, show=False, max_display=20)
        plt.title('SHAP Summary: Most Important Bits for Cipher Detection')
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        print("[create_summary_plot] Summary plot created.")
        return img_base64
    
    def create_feature_importance_plot(self):
        print("[create_feature_importance_plot] Creating SHAP feature importance bar plot.")
        plt.figure(figsize=(12, 6))
        shap.summary_plot(self.shap_values_pos, self.X_test, plot_type="bar", show=False, max_display=15)
        plt.title('Mean |SHAP Value| - Feature Importance')
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        print("[create_feature_importance_plot] Feature importance bar plot created.")
        return img_base64
    
    def create_comparison_plot(self):
        print("[create_comparison_plot] Creating comparison plot between RF and SHAP importances.")
        rf_importance = self.model.feature_importances_
        
        # Use the already processed shap_values_pos
        shap_importance = np.abs(self.shap_values_pos).mean(axis=0)
        top_indices = np.argsort(shap_importance)[-15:]
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(len(top_indices)), rf_importance[top_indices], color='lightblue')
        plt.xticks(range(len(top_indices)), [f'bit_{i}' for i in top_indices], rotation=45)
        plt.title('Random Forest Feature Importance (Top 15)')
        plt.ylabel('RF Importance')

        plt.subplot(1, 2, 2)
        plt.bar(range(len(top_indices)), shap_importance[top_indices], color='lightcoral')
        plt.xticks(range(len(top_indices)), [f'bit_{i}' for i in top_indices], rotation=45)
        plt.title('SHAP Feature Importance (Top 15)')
        plt.ylabel('Mean |SHAP Value|')

        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        print("[create_comparison_plot] Comparison plot created.")
        return img_base64
    
    def analyze_individual_predictions(self, y_pred, y_pred_proba):
        print("[analyze_individual_predictions] Analyzing individual example predictions.")

        y_test_values = self.y_test.values if hasattr(self.y_test, 'values') else np.array(self.y_test)
        y_pred_array = np.array(y_pred)

        correct_cipher = np.where((y_test_values == 1) & (y_pred_array == 1))[0]
        correct_random = np.where((y_test_values == 0) & (y_pred_array == 0))[0]

        examples = []

        if len(correct_cipher) > 0:
            examples.append(("Correctly Identified Cipher", correct_cipher[0]))
        if len(correct_random) > 0:
            examples.append(("Correctly Identified Random", correct_random[0]))

        example_data = []

        for desc, idx in examples[:2]:
            true_label_name = 'Cipher' if y_test_values[idx] == 1 else 'Random'
            predicted_prob = y_pred_proba[idx][1]

            shap_row = self.shap_values_pos[idx]

            # Ensure we are working with a 1D array of scalar SHAP values
            if isinstance(shap_row, (list, np.ndarray)) and shap_row.ndim > 1:
                shap_row = shap_row.flatten()

            feature_contributions = list(zip(self.feature_names, shap_row))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

            top_features = []
            for feature, contrib in feature_contributions[:10]:
                direction = "→ Cipher" if contrib > 0 else "→ Random"
                top_features.append({
                    'feature': feature,
                    'contribution': float(contrib),
                    'direction': direction
                })

            example_data.append({
                'type': desc,
                'true_label': true_label_name,
                'predicted_prob': float(predicted_prob),
                'top_features': top_features
            })

        print("[analyze_individual_predictions] Example analysis complete.")
        return example_data

    def create_confidence_analysis_plot(self, y_pred_proba):
        print("[create_confidence_analysis_plot] Creating plots for confidence analysis.")
        confidence = np.max(y_pred_proba, axis=1)
        
        # Fix: Ensure shap_magnitude has the same size as confidence
        # Calculate magnitude for each test sample
        shap_magnitude = np.abs(self.shap_values_pos).sum(axis=1)
        
        print(f"[DEBUG] confidence shape: {confidence.shape}, shap_magnitude shape: {shap_magnitude.shape}")
        
        # Ensure both arrays have the same length
        min_length = min(len(confidence), len(shap_magnitude))
        confidence = confidence[:min_length]
        shap_magnitude = shap_magnitude[:min_length]
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(confidence, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Distribution of Model Confidence')
        
        plt.subplot(1, 3, 2)
        plt.scatter(confidence, shap_magnitude, alpha=0.6, color='coral')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Total SHAP Magnitude')
        plt.title('Confidence vs SHAP Magnitude')
        
        correlation = np.corrcoef(confidence, shap_magnitude)[0, 1]
        plt.text(0.1, 0.9, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes)
        
        plt.subplot(1, 3, 3)
        # Convert to numpy arrays for safe indexing
        y_test_values = self.y_test.values if hasattr(self.y_test, 'values') else np.array(self.y_test)
        # Ensure y_test_values matches the trimmed confidence array length
        y_test_values = y_test_values[:min_length]
        
        cipher_conf = confidence[y_test_values == 1]
        random_conf = confidence[y_test_values == 0]
        plt.hist(cipher_conf, alpha=0.7, label='Cipher Streams', bins=15, color='red')
        plt.hist(random_conf, alpha=0.7, label='Random Streams', bins=15, color='blue')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence by True Class')
        plt.legend()
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        print(f"[create_confidence_analysis_plot] Plots created. Correlation: {correlation:.3f}")
        return img_base64, correlation, confidence
    
    def create_bit_importance_plot(self):
        print("[create_bit_importance_plot] Calculating bit importance and generating plot.")
        bit_importance = np.abs(self.shap_values_pos).mean(0)
        top_bits = np.argsort(bit_importance)[-10:]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_bits)), bit_importance[top_bits], color='green', alpha=0.7)
        plt.xlabel('Bit Position')
        plt.ylabel('Mean |SHAP Value|')
        plt.title('Top 10 Most Important Bit Positions')
        plt.xticks(range(len(top_bits)), [f'bit_{i}' for i in top_bits], rotation=45)
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        bit_analysis = []
        for i, bit_idx in enumerate(reversed(top_bits), 1):
            # Convert to numpy arrays for safe boolean indexing
            y_values = self.y.values if hasattr(self.y, 'values') else np.array(self.y)
            cipher_mask = (y_values == 1)
            random_mask = (y_values == 0)
            
            avg_cipher = self.X.iloc[cipher_mask, bit_idx].mean()
            avg_random = self.X.iloc[random_mask, bit_idx].mean()
            importance = bit_importance[bit_idx]
            
            bit_analysis.append({
                'rank': i,
                'bit_position': int(bit_idx),
                'importance': float(importance),
                'cipher_avg': float(avg_cipher),
                'random_avg': float(avg_random)
            })
        
        print("[create_bit_importance_plot] Bit importance analysis complete.")
        return img_base64, top_bits, bit_importance, bit_analysis

analyzer = ShapAnalyzer()

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    try:
        print("[API] /api/analyze endpoint hit.")
        print("🚀 Started at:", datetime.now())
        cipher_file = request.files.get('cipher_file')
        random_file = request.files.get('random_file')
        
        if not cipher_file or not random_file:
            print("[API] Error: Both files are required.")
            return jsonify({'error': 'Both files are required'}), 400
        
        print("[API] Processing data...")
        X_train, X_test, y_train, y_test, X, y = analyzer.process_data(cipher_file, random_file)
        
        print("[API] Training model...")
        model = analyzer.train_model(X_train, y_train)
        
        print("[API] Predicting and calculating accuracy...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("[API] Generating SHAP values...")
        shap_values_pos = analyzer.generate_shap_values()
        
        print("[API] Creating plots...")
        summary_plot = analyzer.create_summary_plot()
        importance_plot = analyzer.create_feature_importance_plot()
        comparison_plot = analyzer.create_comparison_plot()
        individual_examples = analyzer.analyze_individual_predictions(y_pred, y_pred_proba)
        confidence_plot, correlation, confidence = analyzer.create_confidence_analysis_plot(y_pred_proba)
        bit_plot, top_bits, bit_importance, detailed_bit_analysis = analyzer.create_bit_importance_plot()
        
        high_conf_threshold = 0.9
        high_conf_predictions = (confidence > high_conf_threshold).sum()
        
        # Fixed the top bit analysis section
        top_bit_analysis = []
        for i, bit_idx in enumerate(reversed(top_bits[-5:]), 1):
            # Convert to numpy arrays for safe boolean indexing
            y_values = y.values if hasattr(y, 'values') else np.array(y)
            avg_cipher = X[y_values == 1].iloc[:, bit_idx].mean()
            avg_random = X[y_values == 0].iloc[:, bit_idx].mean()
            importance = bit_importance[bit_idx]
            top_bit_analysis.append({
                'rank': i,
                'bit_position': int(bit_idx),
                'importance': float(importance),
                'cipher_avg': float(avg_cipher),
                'random_avg': float(avg_random)
            })
        
        print("[API] Assembling results...")
        results = {
            'success': True,
            'accuracy': float(accuracy),
            'total_bits': int(X.shape[1]),
            'cipher_streams': int((y.values if hasattr(y, 'values') else np.array(y) == 1).sum()),
            'random_streams': int((y.values if hasattr(y, 'values') else np.array(y) == 0).sum()),
            'avg_confidence': float(confidence.mean()),
            'confidence_correlation': float(correlation),
            'high_conf_predictions': int(high_conf_predictions),
            'high_conf_percentage': float(100 * high_conf_predictions / len(confidence)),
            'plots': {
                'summary_plot': summary_plot,
                'importance_plot': importance_plot,
                'comparison_plot': comparison_plot,
                'confidence_plot': confidence_plot,
                'bit_plot': bit_plot
            },
            'individual_examples': individual_examples,
            'top_bits': top_bit_analysis,
            'detailed_bit_analysis': detailed_bit_analysis,
            'finished_at': datetime.now().isoformat()
        }
        
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
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    try:
        print("[MAIN] Starting Flask server on 0.0.0.0:5000...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"[MAIN] Failed to start server: {e}")
        import traceback
        traceback.print_exc()