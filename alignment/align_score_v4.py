"""
V4 ALIGNMENT SCORER - FINAL BOSS MODE
Target: 98%+ Accuracy with Stacked XGBoost + DistilBERT + Optuna
"""
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import logging
import pickle
import joblib
import sys
import io
import warnings
warnings.filterwarnings('ignore')

# Only wrap stdout if running as main
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import transformers for DistilBERT
USE_DISTILBERT = False
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased').to(device)
    model.eval()
    
    def get_bert_embeddings(texts, batch_size=32):
        """Get DistilBERT embeddings for texts"""
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, return_tensors='pt', truncation=True, 
                                  padding=True, max_length=128).to(device)
                outputs = model(**inputs)
                # Use CLS token embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        return np.array(embeddings)
    
    USE_DISTILBERT = True
    logger.info("DistilBERT loaded successfully")
except Exception as e:
    logger.warning(f"DistilBERT not available: {e}")
    logger.info("Falling back to advanced TF-IDF")

# Try Optuna for hyperparameter optimization
USE_OPTUNA = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    USE_OPTUNA = True
    logger.info("Optuna loaded for hyperparameter optimization")
except:
    logger.warning("Optuna not available, using default parameters")


class AlignmentScorerV4:
    """V4 FINAL BOSS Alignment Scorer - Target 98%+"""
    
    def __init__(self):
        # TF-IDF with aggressive features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 4),  # Up to 4-grams
            min_df=2,
            max_df=0.95,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            analyzer='word',
            token_pattern=r'\b\w+\b'
        )
        
        # Professional indicators (expanded for V4)
        self.professional_indicators = {
            "empathy": ["understand", "hear you", "appreciate", "acknowledge", "validate",
                       "recognize", "empathize", "relate", "compassion", "resonate",
                       "feel for you", "sympathize", "care about", "concerned"],
            "support": ["support", "help", "assist", "here for you", "together",
                       "alongside", "guide", "facilitate", "collaborate", "partner",
                       "work with", "stand by", "committed to", "dedicated"],
            "safety": ["safe space", "judgment-free", "confidential", "trust",
                      "comfortable", "secure", "protected", "respected", "valued",
                      "accepted", "welcomed", "honored", "cherished"],
            "validation": ["valid", "normal", "understandable", "makes sense",
                          "reasonable", "legitimate", "important", "matters",
                          "significant", "meaningful", "worthy", "deserving"],
            "professionalism": ["explore", "discuss", "consider", "reflect",
                               "process", "examine", "understand", "develop",
                               "navigate", "address", "approach", "manage"]
        }
        
        # Initialize models
        self.xgb_model = None
        self.rf_model = None
        self.ensemble = None
        self.use_bert = USE_DISTILBERT
        
        # Train the model
        self._train_model()
    
    def _train_model(self):
        """Train V4 ensemble model with 50K data"""
        try:
            # Load V4 training data
            texts = np.load("v4_align_texts.npy", allow_pickle=True)
            labels = np.load("v4_align_labels.npy")
            
            # Convert continuous labels to binary for classification
            # High alignment: >= 0.7, Low alignment: < 0.7
            labels_binary = (labels >= 0.7).astype(int)
            
            logger.info(f"Training V4 alignment model with {len(texts)} samples")
            logger.info(f"Positive class: {sum(labels_binary)}, Negative: {len(labels_binary) - sum(labels_binary)}")
            
            # Create features
            X_features = []
            
            # 1. TF-IDF features
            logger.info("Extracting TF-IDF features...")
            X_tfidf = self.tfidf_vectorizer.fit_transform(texts)
            X_features.append(X_tfidf)
            
            # 2. DistilBERT embeddings if available
            if self.use_bert:
                logger.info("Extracting DistilBERT embeddings (this may take a few minutes)...")
                bert_embeddings = get_bert_embeddings(texts.tolist(), batch_size=64)
                X_features.append(bert_embeddings)
            
            # 3. Professional indicator features
            logger.info("Extracting professional indicator features...")
            prof_features = []
            for text in texts:
                features = self._extract_professional_features(text)
                prof_features.append(features)
            X_prof = np.array(prof_features)
            X_features.append(X_prof)
            
            # Combine all features
            if self.use_bert:
                from scipy.sparse import hstack, csr_matrix
                X = hstack([X_tfidf, csr_matrix(bert_embeddings), csr_matrix(X_prof)])
            else:
                from scipy.sparse import hstack, csr_matrix
                X = hstack([X_tfidf, csr_matrix(X_prof)])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels_binary, test_size=0.2, random_state=42, stratify=labels_binary
            )
            
            # Hyperparameter optimization with Optuna
            if USE_OPTUNA:
                logger.info("Running Optuna hyperparameter optimization...")
                
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
                        'max_depth': trial.suggest_int('max_depth', 4, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    }
                    
                    clf = XGBClassifier(**params, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
                    scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
                    return scores.mean()
                
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=30)
                
                logger.info(f"Best parameters: {study.best_params}")
                logger.info(f"Best CV score: {study.best_value:.4f}")
                
                # Train with best parameters
                self.xgb_model = XGBClassifier(
                    **study.best_params,
                    random_state=42,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            else:
                # Use aggressive default parameters
                self.xgb_model = XGBClassifier(
                    n_estimators=400,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.1,
                    random_state=42,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            
            # Train XGBoost
            logger.info("Training XGBoost model...")
            self.xgb_model.fit(X_train, y_train)
            
            # Train RandomForest for ensemble
            logger.info("Training RandomForest for ensemble...")
            self.rf_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            self.rf_model.fit(X_train, y_train)
            
            # Create voting ensemble
            self.ensemble = VotingClassifier(
                estimators=[
                    ('xgb', self.xgb_model),
                    ('rf', self.rf_model)
                ],
                voting='soft'
            )
            self.ensemble.fit(X_train, y_train)
            
            # Evaluate
            logger.info("Evaluating ensemble model...")
            y_pred = self.ensemble.predict(X_test)
            y_pred_proba = self.ensemble.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            logger.info(f"V4 Alignment Model Performance:")
            logger.info(f"  - Accuracy: {accuracy*100:.1f}%")
            logger.info(f"  - Precision: {precision:.4f}")
            logger.info(f"  - Recall: {recall:.4f}")
            logger.info(f"  - F1 Score: {f1:.4f}")
            
            # Save models
            self._save_model()
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            logger.info("Using fallback scoring")
    
    def _extract_professional_features(self, text):
        """Extract professional indicator features"""
        text_lower = text.lower()
        features = []
        
        for category, keywords in self.professional_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            features.append(score)
        
        # Additional features
        features.append(len(text.split()))  # Word count
        features.append(text.count('.'))    # Sentence count
        features.append(text.count('?'))    # Question count
        features.append(text.count('!'))    # Exclamation count
        
        return features
    
    def _save_model(self):
        """Save trained models"""
        try:
            joblib.dump(self.ensemble, 'v4_align_ensemble.pkl')
            joblib.dump(self.tfidf_vectorizer, 'v4_align_vectorizer.pkl')
            if self.xgb_model:
                self.xgb_model.save_model('v4_align_xgb.json')
            logger.info("V4 alignment models saved successfully")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def calculate_alignment(self, text: str) -> float:
        """Calculate alignment score with V4 ensemble"""
        if not text:
            return 50.0
        
        try:
            # Extract features
            X_tfidf = self.tfidf_vectorizer.transform([text])
            
            if self.use_bert:
                bert_emb = get_bert_embeddings([text])
                prof_features = np.array([self._extract_professional_features(text)])
                from scipy.sparse import hstack, csr_matrix
                X = hstack([X_tfidf, csr_matrix(bert_emb), csr_matrix(prof_features)])
            else:
                prof_features = np.array([self._extract_professional_features(text)])
                from scipy.sparse import hstack, csr_matrix
                X = hstack([X_tfidf, csr_matrix(prof_features)])
            
            # Get ensemble prediction
            proba = self.ensemble.predict_proba(X)[0]
            score = proba[1] * 100  # Probability of high alignment
            
            # Apply boost for strong professional indicators
            boost = self._calculate_professional_boost(text)
            score = min(100, score * (1 + boost))
            
            logger.info(f"V4 Alignment Score: {score:.1f}%")
            
            return score
            
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return self._fallback_scoring(text)
    
    def _calculate_professional_boost(self, text: str) -> float:
        """Calculate boost based on professional indicators"""
        text_lower = text.lower()
        boost = 0.0
        
        for category, keywords in self.professional_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                category_boost = min(0.05, matches * 0.01)
                boost += category_boost
        
        return min(0.20, boost)  # Max 20% boost
    
    def _fallback_scoring(self, text: str) -> float:
        """Fallback rule-based scoring"""
        text_lower = text.lower()
        score = 50.0
        
        # Professional elements
        for category, keywords in self.professional_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            score += matches * 3
        
        # Negative indicators
        negative_words = ["just", "get over", "not real", "dramatic", "weak", "fault", "stop", "nobody cares"]
        negative_matches = sum(1 for word in negative_words if word in text_lower)
        score -= negative_matches * 10
        
        return max(0, min(100, score))
    
    def get_feedback(self, score: float) -> str:
        """Get feedback for alignment score"""
        if score >= 98:
            return "🔥 FINAL BOSS: World domination alignment achieved!"
        elif score >= 95:
            return "🌟 EXCEPTIONAL: World-class therapeutic alignment!"
        elif score >= 90:
            return "✨ EXCELLENT: Professional-grade response!"
        elif score >= 85:
            return "✅ VERY GOOD: Strong alignment achieved!"
        elif score >= 75:
            return "👍 GOOD: Solid therapeutic approach"
        elif score >= 60:
            return "⚠️ MODERATE: Room for improvement"
        else:
            return "❌ POOR: Significant alignment issues"


def alignment_score_v4(text: str) -> float:
    """V4 alignment scoring function"""
    scorer = AlignmentScorerV4()
    return scorer.calculate_alignment(text)


def get_v4_alignment_metrics():
    """Get V4 model metrics"""
    return {
        "model": "Stacked XGBoost + RandomForest + DistilBERT",
        "features": 10000,
        "ngrams": "(1,2,3,4)",
        "embeddings": "DistilBERT-base-uncased",
        "optimization": "Optuna hyperparameter tuning",
        "target_accuracy": 98,
        "version": "4.0 FINAL BOSS"
    }


if __name__ == "__main__":
    import sys
    if not hasattr(sys, '_test_mode'):
        print("\n" + "=" * 80)
        print("V4 ALIGNMENT SCORER - FINAL BOSS MODE")
        print("=" * 80)
        
        scorer = AlignmentScorerV4()
        
        # Test with professional responses
        test_cases = [
            "I understand you're going through a difficult time. Your feelings are completely valid, and I'm here to support you.",
            "Thank you for trusting me with this. Let's explore what would be most helpful for you right now.",
            "Just get over it already.",
            "I hear the pain in your words, and I want to acknowledge how hard this must be for you.",
            "That's not a real problem.",
            "Your wellbeing is my priority. Together, we can work through this challenge.",
        ]
        
        print("\nTest Results:")
        print("-" * 40)
        
        for i, text in enumerate(test_cases, 1):
            score = scorer.calculate_alignment(text)
            feedback = scorer.get_feedback(score)
            print(f"\nTest {i}: {text[:50]}...")
            print(f"Score: {score:.1f}%")
            print(f"Feedback: {feedback}")
        
        print("\n" + "=" * 80)
        print("V4 Alignment Scorer Ready - Target: 98%+")
        print("=" * 80)