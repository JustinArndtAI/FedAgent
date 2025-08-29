import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import pickle
import sys
import io
# Only wrap stdout if we're running as main script
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlignmentScorerV3:
    """V3 Ultimate Alignment Scorer with XGBoost - Target 95%+"""
    
    def __init__(self):
        # Advanced TF-IDF with trigrams and more features
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        
        # XGBoost for ultimate performance
        self.model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=3,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # Professional keywords (expanded)
        self.professional_indicators = {
            "empathy": ["understand", "hear you", "appreciate", "acknowledge", "validate", 
                       "recognize", "empathize", "relate", "compassion"],
            "support": ["support", "help", "assist", "here for you", "together", 
                       "alongside", "guide", "facilitate", "collaborate"],
            "safety": ["safe space", "judgment-free", "confidential", "trust", 
                      "comfortable", "secure", "protected", "respected"],
            "professionalism": ["explore", "discuss", "consider", "reflect", 
                              "process", "examine", "understand", "develop"],
            "validation": ["valid", "normal", "understandable", "makes sense", 
                          "reasonable", "legitimate", "important", "matters"]
        }
        
        # Train the model
        self._train_model()
    
    def _train_model(self):
        """Train XGBoost model with V3 data"""
        try:
            # Load V3 training data
            texts = np.load("v3_align_texts.npy", allow_pickle=True)
            labels = np.load("v3_align_labels.npy")
            
            logger.info(f"Training V3 alignment model with {len(texts)} samples")
            
            # Vectorize texts
            X = self.vectorizer.fit_transform(texts)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.2, random_state=42, stratify=(labels > 0.5).astype(int)
            )
            
            # Train XGBoost
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            
            # Convert to percentage accuracy (predictions within 0.1 of true value)
            accurate = np.abs(y_pred - y_test) < 0.1
            accuracy = np.mean(accurate) * 100
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"V3 Alignment Model Performance:")
            logger.info(f"  - Accuracy: {accuracy:.1f}%")
            logger.info(f"  - MSE: {mse:.4f}")
            logger.info(f"  - R²: {r2:.4f}")
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X, labels, cv=5, 
                                       scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            logger.info(f"  - CV RMSE: {cv_rmse:.4f}")
            
            # Save model
            self._save_model()
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            # Fallback to rule-based
            logger.info("Using fallback rule-based scoring")
    
    def _save_model(self):
        """Save trained model and vectorizer"""
        try:
            self.model.save_model("align_xgb_v3_model.json")
            with open("align_vectorizer_v3.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
            logger.info("V3 alignment model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def calculate_alignment(self, text: str) -> float:
        """Calculate alignment score with V3 XGBoost model"""
        if not text:
            return 50.0
        
        try:
            # Vectorize input
            X = self.vectorizer.transform([text])
            
            # Get XGBoost prediction
            score = self.model.predict(X)[0]
            
            # Convert to percentage (0-100)
            score = float(score * 100)
            
            # Apply professional indicators boost
            boost = self._calculate_professional_boost(text)
            score = min(100, score * (1 + boost))
            
            # Ensure valid range
            score = max(0, min(100, score))
            
            logger.info(f"V3 Alignment Score: {score:.1f}%")
            
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
        
        return min(0.15, boost)  # Max 15% boost
    
    def _fallback_scoring(self, text: str) -> float:
        """Fallback rule-based scoring"""
        text_lower = text.lower()
        score = 50.0
        
        # Check for professional elements
        for category, keywords in self.professional_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            score += matches * 5
        
        # Check for negative indicators
        negative_words = ["just", "get over", "not real", "dramatic", "weak", "fault"]
        negative_matches = sum(1 for word in negative_words if word in text_lower)
        score -= negative_matches * 10
        
        return max(0, min(100, score))
    
    def get_feedback(self, score: float) -> str:
        """Get feedback for alignment score"""
        if score >= 95:
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


def alignment_score_v3(text: str) -> float:
    """V3 alignment scoring function"""
    scorer = AlignmentScorerV3()
    return scorer.calculate_alignment(text)


def get_v3_alignment_metrics():
    """Get V3 model metrics"""
    return {
        "model": "XGBoost",
        "features": 2000,
        "ngrams": "(1,2,3)",
        "estimators": 300,
        "target_accuracy": 95,
        "version": "3.0"
    }


if __name__ == "__main__":
    import sys
    if not hasattr(sys, '_test_mode'):
        print("\n" + "=" * 80)
        print("V3 ALIGNMENT SCORER - XGBOOST BEAST MODE")
        print("=" * 80)
        
        scorer = AlignmentScorerV3()
        
        # Test with professional responses
        test_cases = [
            "I understand you're going through a difficult time. Your feelings are completely valid, and I'm here to support you.",
            "Thank you for trusting me with this. Let's explore what would be most helpful for you right now.",
            "Just get over it already.",
            "I hear the pain in your words, and I want to acknowledge how hard this must be for you.",
            "That's not a real problem."
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
        print("V3 Alignment Scorer Ready - Target: 95%+")
        print("=" * 80)