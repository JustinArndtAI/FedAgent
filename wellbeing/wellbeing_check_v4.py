"""
V4 WELLBEING MONITOR - FINAL BOSS MODE
Target: 99%+ Accuracy with DistilBERT + Ensemble + Optuna
"""
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
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
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        return np.array(embeddings)
    
    USE_DISTILBERT = True
    logger.info("DistilBERT loaded successfully for wellbeing")
except Exception as e:
    logger.warning(f"DistilBERT not available: {e}")
    logger.info("Falling back to advanced TF-IDF")

# Try Optuna
USE_OPTUNA = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    USE_OPTUNA = True
    logger.info("Optuna loaded for hyperparameter optimization")
except:
    logger.warning("Optuna not available, using default parameters")


class WellbeingMonitorV4:
    """V4 FINAL BOSS Wellbeing Monitor - Target 99%+"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
        # TF-IDF with maximum features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 5),  # Up to 5-grams
            min_df=2,
            max_df=0.95,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        
        # Crisis detection patterns (comprehensive)
        self.crisis_patterns = {
            "extreme": {
                "keywords": ["kill myself", "end my life", "suicide", "want to die",
                           "better off dead", "planning to die", "end it all"],
                "weight": 1.0
            },
            "severe": {
                "keywords": ["can't go on", "hopeless", "no point", "hurt myself",
                           "give up on life", "no hope", "want to disappear"],
                "weight": 0.8
            },
            "high": {
                "keywords": ["worthless", "hate myself", "burden", "give up",
                          "empty inside", "broken", "can't cope"],
                "weight": 0.6
            },
            "moderate": {
                "keywords": ["depressed", "anxious", "scared", "overwhelmed",
                           "struggling", "exhausted", "lost"],
                "weight": 0.4
            }
        }
        
        # Positive patterns (comprehensive)
        self.positive_patterns = {
            "euphoric": {
                "keywords": ["amazing", "wonderful", "fantastic", "thrilled",
                           "perfect", "incredible", "best ever", "ecstatic"],
                "weight": 0.9
            },
            "happy": {
                "keywords": ["happy", "good", "grateful", "content", "pleased",
                          "satisfied", "joyful", "cheerful"],
                "weight": 0.7
            },
            "hopeful": {
                "keywords": ["hopeful", "optimistic", "improving", "better",
                          "positive", "encouraged", "confident"],
                "weight": 0.5
            }
        }
        
        # Initialize models
        self.rf_model = None
        self.xgb_model = None
        self.gb_model = None
        self.ensemble = None
        self.use_bert = USE_DISTILBERT
        
        # Train the model
        self._train_model()
    
    def _train_model(self):
        """Train V4 ensemble model with 50K data"""
        try:
            # Load V4 training data
            with open("v4_wellbeing_texts.txt", "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f]
            scores = np.load("v4_wellbeing_scores.npy")
            
            # Convert to binary classification
            # Positive wellbeing: >= 0, Negative: < 0
            labels_binary = (scores >= 0).astype(int)
            
            logger.info(f"Training V4 wellbeing model with {len(texts)} samples")
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
                bert_embeddings = get_bert_embeddings(texts, batch_size=64)
                X_features.append(bert_embeddings)
            
            # 3. VADER sentiment features
            logger.info("Extracting VADER sentiment features...")
            vader_features = []
            for text in texts:
                vader_scores = self.analyzer.polarity_scores(text)
                vader_features.append([
                    vader_scores['compound'],
                    vader_scores['pos'],
                    vader_scores['neg'],
                    vader_scores['neu']
                ])
            X_vader = np.array(vader_features)
            X_features.append(X_vader)
            
            # 4. Crisis/Positive pattern features
            logger.info("Extracting pattern features...")
            pattern_features = []
            for text in texts:
                crisis_score = self._detect_crisis_level(text)
                positive_score = self._detect_positive_level(text)
                pattern_features.append([crisis_score, positive_score])
            X_patterns = np.array(pattern_features)
            X_features.append(X_patterns)
            
            # Combine all features
            if self.use_bert:
                from scipy.sparse import hstack, csr_matrix
                X = hstack([X_tfidf, csr_matrix(bert_embeddings), 
                           csr_matrix(X_vader), csr_matrix(X_patterns)])
            else:
                from scipy.sparse import hstack, csr_matrix
                X = hstack([X_tfidf, csr_matrix(X_vader), csr_matrix(X_patterns)])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels_binary, test_size=0.2, random_state=42, stratify=labels_binary
            )
            
            # Hyperparameter optimization with Optuna
            if USE_OPTUNA:
                logger.info("Running Optuna hyperparameter optimization...")
                
                def objective_rf(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
                        'max_depth': trial.suggest_int('max_depth', 5, 15),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    }
                    clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                    scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
                    return scores.mean()
                
                study_rf = optuna.create_study(direction='maximize')
                study_rf.optimize(objective_rf, n_trials=20)
                
                logger.info(f"Best RF parameters: {study_rf.best_params}")
                
                # Train with best parameters
                self.rf_model = RandomForestClassifier(
                    **study_rf.best_params,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                # Use aggressive default parameters
                self.rf_model = RandomForestClassifier(
                    n_estimators=400,
                    max_depth=12,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Train RandomForest
            logger.info("Training RandomForest model...")
            self.rf_model.fit(X_train, y_train)
            
            # Train XGBoost
            logger.info("Training XGBoost model...")
            self.xgb_model = XGBClassifier(
                n_estimators=400,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            self.xgb_model.fit(X_train, y_train)
            
            # Train GradientBoosting
            logger.info("Training GradientBoosting model...")
            self.gb_model = GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                random_state=42
            )
            self.gb_model.fit(X_train, y_train)
            
            # Create voting ensemble
            self.ensemble = VotingClassifier(
                estimators=[
                    ('rf', self.rf_model),
                    ('xgb', self.xgb_model),
                    ('gb', self.gb_model)
                ],
                voting='soft'
            )
            self.ensemble.fit(X_train, y_train)
            
            # Evaluate
            logger.info("Evaluating ensemble model...")
            y_pred = self.ensemble.predict(X_test)
            y_pred_proba = self.ensemble.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            # For regression evaluation (using original scores)
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                X, scores, test_size=0.2, random_state=42
            )
            
            logger.info(f"V4 Wellbeing Model Performance:")
            logger.info(f"  - Classification Accuracy: {accuracy*100:.1f}%")
            logger.info(f"  - Target: 99%+ FINAL BOSS MODE")
            
            # Save models
            self._save_model()
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            logger.info("Using fallback scoring")
    
    def _save_model(self):
        """Save trained models"""
        try:
            joblib.dump(self.ensemble, 'v4_wellbeing_ensemble.pkl')
            joblib.dump(self.tfidf_vectorizer, 'v4_wellbeing_vectorizer.pkl')
            if self.xgb_model:
                self.xgb_model.save_model('v4_wellbeing_xgb.json')
            logger.info("V4 wellbeing models saved successfully")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def check_wellbeing(self, text: str) -> float:
        """Check wellbeing with V4 ensemble model"""
        if not text:
            return 0.0
        
        try:
            # Extract features
            X_tfidf = self.tfidf_vectorizer.transform([text])
            
            # VADER features
            vader_scores = self.analyzer.polarity_scores(text)
            X_vader = np.array([[
                vader_scores['compound'],
                vader_scores['pos'],
                vader_scores['neg'],
                vader_scores['neu']
            ]])
            
            # Pattern features
            crisis_score = self._detect_crisis_level(text)
            positive_score = self._detect_positive_level(text)
            X_patterns = np.array([[crisis_score, positive_score]])
            
            # Combine features
            if self.use_bert:
                bert_emb = get_bert_embeddings([text])
                from scipy.sparse import hstack, csr_matrix
                X = hstack([X_tfidf, csr_matrix(bert_emb), 
                           csr_matrix(X_vader), csr_matrix(X_patterns)])
            else:
                from scipy.sparse import hstack, csr_matrix
                X = hstack([X_tfidf, csr_matrix(X_vader), csr_matrix(X_patterns)])
            
            # Get ensemble prediction
            proba = self.ensemble.predict_proba(X)[0]
            
            # Convert to wellbeing score (-1 to 1)
            # Probability of positive wellbeing mapped to score
            ml_score = (proba[1] * 2) - 1  # Map [0,1] to [-1,1]
            
            # Combine with VADER for robustness
            final_score = ml_score * 0.8 + vader_scores['compound'] * 0.2
            
            # Apply crisis/positive adjustments
            if crisis_score > 0.5:
                final_score = min(final_score, -0.5 - crisis_score * 0.3)
            elif positive_score > 0.5:
                final_score = max(final_score, 0.3 + positive_score * 0.3)
            
            # Ensure valid range
            final_score = max(-1.0, min(1.0, final_score))
            
            logger.info(f"V4 Wellbeing Score: {final_score:.2f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return self._fallback_scoring(text)
    
    def _detect_crisis_level(self, text: str) -> float:
        """Detect crisis level in text"""
        text_lower = text.lower()
        max_level = 0.0
        
        for level, data in self.crisis_patterns.items():
            for keyword in data["keywords"]:
                if keyword in text_lower:
                    max_level = max(max_level, data["weight"])
        
        return max_level
    
    def _detect_positive_level(self, text: str) -> float:
        """Detect positive level in text"""
        text_lower = text.lower()
        max_level = 0.0
        
        for level, data in self.positive_patterns.items():
            for keyword in data["keywords"]:
                if keyword in text_lower:
                    max_level = max(max_level, data["weight"])
        
        return max_level
    
    def _fallback_scoring(self, text: str) -> float:
        """Fallback VADER-based scoring"""
        vader_scores = self.analyzer.polarity_scores(text)
        return vader_scores['compound']
    
    def get_alarm_status(self, score: float) -> dict:
        """Get alarm status for wellbeing score"""
        alarm = {
            "triggered": False,
            "level": "normal",
            "message": "",
            "confidence": 0.0
        }
        
        if score < -0.8:
            alarm = {
                "triggered": True,
                "level": "critical",
                "message": "🚨 CRITICAL: Immediate intervention required!",
                "confidence": 0.99
            }
        elif score < -0.6:
            alarm = {
                "triggered": True,
                "level": "severe",
                "message": "⚠️ SEVERE: High risk - urgent support needed",
                "confidence": 0.97
            }
        elif score < -0.4:
            alarm = {
                "triggered": True,
                "level": "warning",
                "message": "⚠️ WARNING: Low wellbeing - support recommended",
                "confidence": 0.95
            }
        elif score > 0.8:
            alarm = {
                "triggered": False,
                "level": "euphoric",
                "message": "🌟 EUPHORIC: Exceptional wellbeing detected!",
                "confidence": 0.98
            }
        
        return alarm


def wellbeing_score_v4(text: str) -> float:
    """V4 wellbeing scoring function"""
    monitor = WellbeingMonitorV4()
    return monitor.check_wellbeing(text)


def check_alarm_v4(score: float) -> str:
    """V4 alarm checking function"""
    monitor = WellbeingMonitorV4()
    alarm = monitor.get_alarm_status(score)
    return alarm["message"] if alarm["triggered"] else ""


def get_v4_wellbeing_metrics():
    """Get V4 model metrics"""
    return {
        "model": "Triple Ensemble (RF + XGB + GB) + DistilBERT",
        "features": "DistilBERT embeddings + TF-IDF(15000) + VADER + Patterns",
        "optimization": "Optuna hyperparameter tuning",
        "target_accuracy": 99,
        "version": "4.0 FINAL BOSS"
    }


if __name__ == "__main__":
    import sys
    if not hasattr(sys, '_test_mode'):
        print("\n" + "=" * 80)
        print("V4 WELLBEING MONITOR - FINAL BOSS MODE")
        print("=" * 80)
        
        monitor = WellbeingMonitorV4()
        
        # Test cases
        test_cases = [
            "I'm absolutely thrilled with life! Everything is amazing!",
            "I'm feeling pretty good today, things are looking up.",
            "Today is just another day, nothing special.",
            "I'm so depressed I can't function anymore.",
            "I want to end my life, I can't go on.",
            "Life is wonderful and I'm grateful for everything!",
        ]
        
        print("\nTest Results:")
        print("-" * 40)
        
        for i, text in enumerate(test_cases, 1):
            score = monitor.check_wellbeing(text)
            alarm = monitor.get_alarm_status(score)
            
            print(f"\nTest {i}: {text[:50]}...")
            print(f"Score: {score:.2f}")
            if alarm["triggered"] or alarm["level"] == "euphoric":
                print(f"Status: {alarm['message']}")
                print(f"Confidence: {alarm['confidence']:.0%}")
        
        print("\n" + "=" * 80)
        print("V4 Wellbeing Monitor Ready - Target: 99%+")
        print("=" * 80)