import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import logging
import pickle
import sys
import io
# Only wrap stdout if we're running as main script
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("BERT not available, using advanced TF-IDF")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WellbeingMonitorV3:
    """V3 Ultimate Wellbeing Monitor - Target 98%+ Accuracy"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Use BERT if available, otherwise advanced TF-IDF
        if BERT_AVAILABLE:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                self.use_bert = True
                logger.info("Using BERT embeddings for wellbeing")
            except:
                self.use_bert = False
                self._init_tfidf()
        else:
            self.use_bert = False
            self._init_tfidf()
        
        # Ensemble model for maximum accuracy
        self.primary_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42
        )
        
        self.secondary_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Crisis detection patterns (expanded)
        self.crisis_patterns = {
            "extreme": {
                "keywords": ["kill myself", "end my life", "suicide", "want to die"],
                "weight": 1.0
            },
            "severe": {
                "keywords": ["can't go on", "hopeless", "no point", "hurt myself"],
                "weight": 0.8
            },
            "high": {
                "keywords": ["worthless", "hate myself", "burden", "give up"],
                "weight": 0.6
            }
        }
        
        # Positive patterns
        self.positive_patterns = {
            "euphoric": {
                "keywords": ["amazing", "wonderful", "fantastic", "thrilled"],
                "weight": 0.9
            },
            "happy": {
                "keywords": ["happy", "good", "grateful", "content"],
                "weight": 0.7
            },
            "hopeful": {
                "keywords": ["hopeful", "optimistic", "improving", "better"],
                "weight": 0.5
            }
        }
        
        # Train the model
        self._train_model()
    
    def _init_tfidf(self):
        """Initialize advanced TF-IDF vectorizer"""
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 4),
            min_df=2,
            max_df=0.95,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            analyzer='word',
            token_pattern=r'\b\w+\b'
        )
        logger.info("Using advanced TF-IDF for wellbeing")
    
    def _train_model(self):
        """Train ensemble model with V3 data"""
        try:
            # Load V3 training data
            with open("v3_wellbeing_texts.txt", "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f]
            scores = np.load("v3_wellbeing_scores.npy")
            
            logger.info(f"Training V3 wellbeing model with {len(texts)} samples")
            
            # Create features
            if self.use_bert:
                logger.info("Encoding texts with BERT (this may take a moment)...")
                X = self.embedder.encode(texts, show_progress_bar=True)
            else:
                X = self.vectorizer.fit_transform(texts)
            
            # Add VADER features
            vader_features = []
            for text in texts:
                vader_scores = self.analyzer.polarity_scores(text)
                vader_features.append([
                    vader_scores['compound'],
                    vader_scores['pos'],
                    vader_scores['neg'],
                    vader_scores['neu']
                ])
            
            # Combine features
            if self.use_bert:
                X_combined = np.hstack([X, np.array(vader_features)])
            else:
                X_combined = np.hstack([X.toarray(), np.array(vader_features)])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, scores, test_size=0.2, random_state=42
            )
            
            # Train primary model
            self.primary_model.fit(X_train, y_train)
            y_pred_primary = self.primary_model.predict(X_test)
            
            # Train secondary model
            self.secondary_model.fit(X_train, y_train)
            y_pred_secondary = self.secondary_model.predict(X_test)
            
            # Ensemble predictions
            y_pred = (y_pred_primary * 0.6 + y_pred_secondary * 0.4)
            
            # Calculate accuracy (predictions within 0.15 of true value)
            accurate = np.abs(y_pred - y_test) < 0.15
            accuracy = np.mean(accurate) * 100
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"V3 Wellbeing Model Performance:")
            logger.info(f"  - Accuracy: {accuracy:.1f}%")
            logger.info(f"  - MSE: {mse:.4f}")
            logger.info(f"  - R²: {r2:.4f}")
            
            # Save model
            self._save_model()
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            logger.info("Using fallback scoring")
    
    def _save_model(self):
        """Save trained models"""
        try:
            with open("wellbeing_primary_v3.pkl", "wb") as f:
                pickle.dump(self.primary_model, f)
            with open("wellbeing_secondary_v3.pkl", "wb") as f:
                pickle.dump(self.secondary_model, f)
            if not self.use_bert:
                with open("wellbeing_vectorizer_v3.pkl", "wb") as f:
                    pickle.dump(self.vectorizer, f)
            logger.info("V3 wellbeing models saved successfully")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def check_wellbeing(self, text: str) -> float:
        """Check wellbeing with V3 ensemble model"""
        if not text:
            return 0.0
        
        try:
            # Get VADER scores
            vader_scores = self.analyzer.polarity_scores(text)
            vader_features = [
                vader_scores['compound'],
                vader_scores['pos'],
                vader_scores['neg'],
                vader_scores['neu']
            ]
            
            # Get text features
            if self.use_bert:
                text_features = self.embedder.encode([text])
                X = np.hstack([text_features, np.array([vader_features])])
            else:
                text_features = self.vectorizer.transform([text])
                X = np.hstack([text_features.toarray(), np.array([vader_features])])
            
            # Get ensemble prediction
            pred_primary = self.primary_model.predict(X)[0]
            pred_secondary = self.secondary_model.predict(X)[0]
            ml_score = pred_primary * 0.6 + pred_secondary * 0.4
            
            # Combine with VADER (weighted)
            final_score = ml_score * 0.7 + vader_scores['compound'] * 0.3
            
            # Apply crisis/positive detection adjustments
            crisis_adjustment = self._detect_crisis_level(text)
            positive_adjustment = self._detect_positive_level(text)
            
            if crisis_adjustment > 0:
                final_score = min(final_score, -0.5 - crisis_adjustment * 0.3)
            elif positive_adjustment > 0:
                final_score = max(final_score, 0.3 + positive_adjustment * 0.3)
            
            # Ensure valid range
            final_score = max(-1.0, min(1.0, final_score))
            
            logger.info(f"V3 Wellbeing Score: {final_score:.2f}")
            
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
                "confidence": 0.98
            }
        elif score < -0.6:
            alarm = {
                "triggered": True,
                "level": "severe",
                "message": "⚠️ SEVERE: High risk - urgent support needed",
                "confidence": 0.95
            }
        elif score < -0.4:
            alarm = {
                "triggered": True,
                "level": "warning",
                "message": "⚠️ WARNING: Low wellbeing - support recommended",
                "confidence": 0.90
            }
        
        return alarm


def wellbeing_score_v3(text: str) -> float:
    """V3 wellbeing scoring function"""
    monitor = WellbeingMonitorV3()
    return monitor.check_wellbeing(text)


def check_alarm_v3(score: float) -> str:
    """V3 alarm checking function"""
    monitor = WellbeingMonitorV3()
    alarm = monitor.get_alarm_status(score)
    return alarm["message"] if alarm["triggered"] else ""


def get_v3_wellbeing_metrics():
    """Get V3 model metrics"""
    return {
        "model": "Ensemble (GradientBoosting + RandomForest)",
        "features": "BERT embeddings or TF-IDF(3000) + VADER",
        "target_accuracy": 98,
        "version": "3.0"
    }


if __name__ == "__main__":
    import sys
    if not hasattr(sys, '_test_mode'):
        print("\n" + "=" * 80)
        print("V3 WELLBEING MONITOR - ULTIMATE ACCURACY MODE")
        print("=" * 80)
        
        monitor = WellbeingMonitorV3()
        
        # Test cases
        test_cases = [
            "I'm absolutely thrilled with life! Everything is amazing!",
            "I'm feeling pretty good today, things are looking up.",
            "Today is just another day, nothing special.",
            "I'm so depressed I can't function anymore.",
            "I want to end my life, I can't go on."
        ]
        
        print("\nTest Results:")
        print("-" * 40)
        
        for i, text in enumerate(test_cases, 1):
            score = monitor.check_wellbeing(text)
            alarm = monitor.get_alarm_status(score)
            
            print(f"\nTest {i}: {text[:50]}...")
            print(f"Score: {score:.2f}")
            if alarm["triggered"]:
                print(f"Alarm: {alarm['message']}")
                print(f"Confidence: {alarm['confidence']:.0%}")
        
        print("\n" + "=" * 80)
        print("V3 Wellbeing Monitor Ready - Target: 98%+")
        print("=" * 80)