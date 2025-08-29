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
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model storage
models = {}

def load_models():
    """Load V6 FINAL models"""
    global models
    logger.info("Loading V6 FINAL BOSS models...")
    
    try:
        # Check which models exist
        if os.path.exists("v6_final_align_ensemble.pkl"):
            model_prefix = "v6_final"
        else:
            model_prefix = "v6"
            
        # Load config
        config_file = f"{model_prefix}_config.json"
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
        else:
            config = {"distilbert_dims": 100}
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load DistilBERT
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
        distilbert.eval()
        
        # Load models
        models = {
            'config': config,
            'device': device,
            'tokenizer': tokenizer,
            'distilbert': distilbert,
            'align_ensemble': joblib.load(f'{model_prefix}_align_ensemble.pkl'),
            'wellbeing_ensemble': joblib.load(f'{model_prefix}_wellbeing_ensemble.pkl'),
            'align_vectorizer': joblib.load('v4_align_vectorizer.pkl'),  # V6 uses V4 vectorizers
            'wellbeing_vectorizer': joblib.load('v4_wellbeing_vectorizer.pkl'),
            'v4_align': joblib.load('v4_align_ensemble.pkl'),
            'v4_wellbeing': joblib.load('v4_wellbeing_ensemble.pkl')
        }
        
        logger.info(f"✅ Models loaded: {model_prefix}")
        return True
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False

def get_distilbert_embedding(text):
    """Extract DistilBERT embedding"""
    inputs = models['tokenizer']([text], padding=True, truncation=True, 
                                  max_length=128, return_tensors='pt').to(models['device'])
    with torch.no_grad():
        outputs = models['distilbert'](**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    return embedding[:models['config'].get('distilbert_dims', 100)]

def predict_single(text):
    """Make predictions for single text"""
    try:
        # Get V4 predictions
        tfidf_align = models['align_vectorizer'].transform([text])
        tfidf_wellbeing = models['wellbeing_vectorizer'].transform([text])
        
        v4_align_pred = models['v4_align'].predict_proba(tfidf_align)[0]
        v4_wellbeing_pred = models['v4_wellbeing'].predict_proba(tfidf_wellbeing)[0]
        
        # Get DistilBERT embedding
        bert_emb = get_distilbert_embedding(text)
        
        # Create meta-features
        X_align = np.concatenate([v4_align_pred, bert_emb]).reshape(1, -1)
        X_wellbeing = np.concatenate([v4_wellbeing_pred, bert_emb]).reshape(1, -1)
        
        # V6 predictions
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
        'version': '6.0 FINAL',
        'architecture': 'V4 + DistilBERT Meta-Learning'
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
            return jsonify({'error': 'Empty text'}), 400
        
        result = predict_single(text)
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        result['processing_time_ms'] = round((time.time() - start_time) * 1000, 2)
        result['model_version'] = 'V6 FINAL BOSS'
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
        if len(texts) > 100:
            return jsonify({'error': 'Maximum 100 texts'}), 400
        
        results = []
        for text in texts:
            result = predict_single(text)
            if result:
                results.append(result)
        
        return jsonify({
            'predictions': results,
            'total': len(results),
            'processing_time_ms': round((time.time() - start_time) * 1000, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Get model metrics"""
    try:
        # Try V6 final report first
        if os.path.exists("v6_final_report.json"):
            with open("v6_final_report.json", "r") as f:
                report = json.load(f)
        elif os.path.exists("v6_training_report.json"):
            with open("v6_training_report.json", "r") as f:
                report = json.load(f)
        else:
            return jsonify({'error': 'Metrics not available'}), 404
        return jsonify(report)
    except:
        return jsonify({'error': 'Metrics error'}), 500

if __name__ == '__main__':
    if load_models():
        print("\n" + "=" * 60)
        print("V6 FINAL BOSS API READY")
        print("=" * 60)
        print("\nEndpoints:")
        print("  POST /predict - Single prediction")
        print("  POST /batch - Batch prediction")
        print("  GET /metrics - Performance metrics")
        print("  GET / - Health check")
        print("\nStarting on http://localhost:5000")
        print("=" * 60 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to load models. Run train_v6_final_50k.py first.")
