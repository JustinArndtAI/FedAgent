#!/usr/bin/env python
"""
V6 FINAL BOSS - PRODUCTION API
EdgeFedAlign Alignment & Wellbeing Classifier
"""
from flask import Flask, request, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
import numpy as np
import json
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model storage
models = {}

def load_models():
    """Load all V6 models and components"""
    global models
    logger.info("Loading V6 FINAL BOSS models...")
    
    try:
        # Load config
        with open("v6_model_config.json", "r") as f:
            config = json.load(f)
        
        # Load DistilBERT
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
        distilbert.eval()
        
        # Load ensembles and vectorizers
        models = {
            'config': config,
            'device': device,
            'tokenizer': tokenizer,
            'distilbert': distilbert,
            'align_ensemble': joblib.load('v6_align_ensemble.pkl'),
            'wellbeing_ensemble': joblib.load('v6_wellbeing_ensemble.pkl'),
            'align_vectorizer': joblib.load('v6_align_vectorizer.pkl'),
            'wellbeing_vectorizer': joblib.load('v6_wellbeing_vectorizer.pkl')
        }
        logger.info("✅ Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False

def get_distilbert_embedding(text):
    """Extract DistilBERT embedding for single text"""
    inputs = models['tokenizer']([text], padding=True, truncation=True, 
                                  max_length=128, return_tensors='pt').to(models['device'])
    
    with torch.no_grad():
        outputs = models['distilbert'](**inputs)
    
    # Mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding

def predict_single(text):
    """Make predictions for single text"""
    try:
        # Get DistilBERT embedding
        db_embedding = get_distilbert_embedding(text)
        
        # Get TF-IDF features
        tfidf_align = models['align_vectorizer'].transform([text]).toarray()
        tfidf_wellbeing = models['wellbeing_vectorizer'].transform([text]).toarray()
        
        # Create hybrid features
        X_align = np.hstack((db_embedding, tfidf_align))
        X_wellbeing = np.hstack((db_embedding, tfidf_wellbeing))
        
        # Predict
        align_prob = models['align_ensemble'].predict_proba(X_align)[0][1]
        wellbeing_prob = models['wellbeing_ensemble'].predict_proba(X_wellbeing)[0][1]
        
        return {
            'alignment_score': float(align_prob),
            'wellbeing_score': float(wellbeing_prob),
            'alignment_label': 'high' if align_prob >= 0.7 else 'low',
            'wellbeing_label': 'positive' if wellbeing_prob >= 0.5 else 'negative'
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'model': 'V6 FINAL BOSS',
        'version': '6.0',
        'features': 'DistilBERT + TF-IDF Hybrid'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    start_time = time.time()
    
    try:
        data = request.json
        
        if 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400
        
        text = data['text']
        
        if not text or len(text.strip()) == 0:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Make prediction
        result = predict_single(text)
        
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Add metadata
        result['processing_time_ms'] = round((time.time() - start_time) * 1000, 2)
        result['model_version'] = 'V6 FINAL BOSS'
        result['text_length'] = len(text)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    start_time = time.time()
    
    try:
        data = request.json
        
        if 'texts' not in data:
            return jsonify({'error': 'Missing "texts" field'}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({'error': 'texts must be a list'}), 400
        
        if len(texts) > 100:
            return jsonify({'error': 'Maximum 100 texts per batch'}), 400
        
        results = []
        for text in texts:
            result = predict_single(text)
            if result:
                results.append(result)
        
        return jsonify({
            'predictions': results,
            'total': len(results),
            'processing_time_ms': round((time.time() - start_time) * 1000, 2),
            'model_version': 'V6 FINAL BOSS'
        })
    
    except Exception as e:
        logger.error(f"Batch API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Get model metrics"""
    try:
        with open("v6_training_report.json", "r") as f:
            report = json.load(f)
        return jsonify(report)
    except:
        return jsonify({'error': 'Metrics not available'}), 404

if __name__ == '__main__':
    # Load models on startup
    if load_models():
        print("\n" + "=" * 60)
        print("🔥 V6 FINAL BOSS API READY 🔥")
        print("=" * 60)
        print("\nEndpoints:")
        print("  POST /predict - Single text prediction")
        print("  POST /batch - Batch prediction (max 100)")
        print("  GET /metrics - Model performance metrics")
        print("  GET / - Health check")
        print("\nStarting server on http://localhost:5000")
        print("=" * 60 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to load models. Please run train_v6_final_boss.py first.")