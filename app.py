"""
Flask API for Flight Cargo Prediction App
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
import datetime
from data_processor import DataProcessor
from model_trainer import ModelTrainer
try:
    from shap_explainer import SHAPExplainer
    SHAP_AVAILABLE = True
except (ImportError, OSError) as e:
    SHAP_AVAILABLE = False
    print(f"Warning: SHAP not available: {e}")
from rule_extractor import RuleExtractor
from generate_sample_data import generate_sample_data
from flask import send_file

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize components
processor = DataProcessor()
trainer = ModelTrainer()
shap_explainer = None
rule_extractor = RuleExtractor()

# Global state
trained = False
current_data = None
current_predictions = None

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/upload', methods=['POST'])
def upload_data():
    """Upload and process CSV data"""
    global current_data, trained
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Load and preprocess data
        df = processor.load_data(file_path)
        current_data = df
        
        # Get data info
        info = {
            'rows': len(df),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'sample': df.head(10).to_dict('records'),
            'statistics': df.describe().to_dict()
        }
        
        return jsonify({
            'success': True,
            'info': info,
            'message': f'Data loaded successfully: {len(df)} rows, {len(df.columns)} columns'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_models():
    """Train ML models on uploaded data"""
    global trained, shap_explainer, rule_extractor, current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded. Please upload CSV first.'}), 400
    
    try:
        # Preprocess data
        X, y = processor.preprocess(current_data, is_training=True)
        
        # Split data
        X_train, X_test, y_train, y_test = processor.split_data(X, y)
        
        # Train models
        trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Evaluate models
        scores = trainer.evaluate_models(X_test, y_test)
        
        # Get best model
        best_name, best_model = trainer.get_best_model()
        
        # Initialize SHAP explainer with best model (if available)
        if SHAP_AVAILABLE:
            shap_explainer = SHAPExplainer(best_model, processor.feature_columns)
            shap_explainer.create_explainer(X_train[:100])  # Use sample for background
        else:
            shap_explainer = None
        
        # Train rule extractor
        rule_extractor.train(X_train.values, y_train.values, processor.feature_columns)
        
        # Save models and processor
        trainer.save_models()
        processor.save_processor('models/processor.pkl')
        
        trained = True
        
        return jsonify({
            'success': True,
            'scores': scores,
            'best_model': best_name,
            'message': 'Models trained successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions on new data"""
    global current_predictions, trained
    
    if not trained:
        return jsonify({'error': 'Models not trained. Please train models first.'}), 400
    
    try:
        data = request.json
        if 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        
        # Preprocess
        X = processor.preprocess(df, is_training=False)
        
        # Predict
        predictions = trainer.predict(X)
        current_predictions = predictions.tolist()
        
        return jsonify({
            'success': True,
            'predictions': current_predictions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights', methods=['GET'])
def get_insights():
    """Get comprehensive insights: feature importance, interactions, rules"""
    global trained, shap_explainer, rule_extractor, current_data
    
    if not trained:
        return jsonify({'error': 'Models not trained. Please train models first.'}), 400
    
    try:
        # Preprocess current data
        X, y = processor.preprocess(current_data, is_training=True)
        X_train, X_test, y_train, y_test = processor.split_data(X, y)
        
        # Get rule-based insights
        rule_insights = rule_extractor.generate_insights(
            X_train.values, y_train.values, processor.feature_columns
        )
        
        # Get SHAP insights if available
        if SHAP_AVAILABLE and shap_explainer is not None:
            shap_insights = shap_explainer.get_insights(X_test[:50], trainer.predict(X_test[:50]))
            feature_importance = shap_insights['feature_importance']
            feature_interactions = shap_insights['feature_interactions']
        else:
            # Fallback: Use model's feature_importances_ if available
            best_name, best_model = trainer.get_best_model()
            if hasattr(best_model, 'feature_importances_'):
                importance = best_model.feature_importances_
                feature_importance = [
                    {'feature': feat, 'importance': float(imp)} 
                    for feat, imp in zip(processor.feature_columns, importance)
                ]
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            else:
                feature_importance = []
            feature_interactions = []
        
        # Combine insights
        insights = {
            'feature_importance': feature_importance,
            'feature_interactions': feature_interactions,
            'rules': rule_insights['rules'],
            'conditional_probabilities': rule_insights['conditional_probabilities'],
            'feature_combinations': rule_insights['feature_combinations']
        }
        
        return jsonify({
            'success': True,
            'insights': insights
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conditional-probability', methods=['POST'])
def get_conditional_probability():
    """Get conditional probability for specific feature ranges"""
    global trained, rule_extractor, current_data
    
    if not trained:
        return jsonify({'error': 'Models not trained. Please train models first.'}), 400
    
    try:
        data = request.json
        feature_ranges = data.get('feature_ranges', {})
        
        # Preprocess current data
        X, y = processor.preprocess(current_data, is_training=True)
        
        # Get conditional probability
        result = rule_extractor.get_conditional_probabilities(
            X.values, y.values, feature_ranges
        )
        
        if result is None:
            return jsonify({
                'success': False,
                'message': 'No data matches the specified conditions'
            }), 400
        
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-sample', methods=['POST'])
def generate_sample():
    """Generate sample flight data"""
    try:
        os.makedirs('data', exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'sample_flight_data_{timestamp}.csv'
        output_path = os.path.join('data', output_file)
        
        # Generate data
        generate_sample_data(n_samples=1000, output_file=output_file)
        
        return jsonify({
            'success': True,
            'message': f'Sample data generated: {output_file}',
            'filename': output_file
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data', methods=['GET'])
def list_data_files():
    """List available CSV files in the data directory"""
    try:
        os.makedirs('data', exist_ok=True)
        files = []
        for f in os.listdir('data'):
            if f.endswith('.csv'):
                path = os.path.join('data', f)
                stats = os.stat(path)
                files.append({
                    'name': f,
                    'size': stats.st_size,
                    'modified': datetime.datetime.fromtimestamp(stats.st_mtime).isoformat()
                })
        
        # Sort by modified time (newest first)
        files.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            'success': True,
            'files': files
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/select', methods=['POST'])
def select_data_file():
    """Select a data file for training/processing"""
    global current_data
    
    try:
        data = request.json
        filename = data.get('filename')
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        file_path = os.path.join('data', filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        # Load and preprocess data
        df = processor.load_data(file_path)
        current_data = df
        
        # Get data info
        info = {
            'rows': len(df),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'sample': df.head(10).to_dict('records'),
            'statistics': df.describe().to_dict()
        }
        
        return jsonify({
            'success': True,
            'info': info,
            'message': f'Data loaded successfully: {len(df)} rows'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/preview/<filename>', methods=['GET'])
def preview_data_file(filename):
    """Preview a specific data file"""
    try:
        file_path = os.path.join('data', filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        df = pd.read_csv(file_path)
        
        info = {
            'rows': len(df),
            'columns': list(df.columns),
            'sample': df.head(10).to_dict('records')
        }
        
        return jsonify({
            'success': True,
            'info': info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current application status"""
    return jsonify({
        'trained': trained,
        'has_data': current_data is not None,
        'data_rows': len(current_data) if current_data is not None else 0,
        'best_model': trainer.get_best_model()[0] if trained else None
    })

# Initialize directories
os.makedirs('static', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')


