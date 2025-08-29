from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import logging
from typing import Dict, Any, List
import re
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WellbeingMonitorV2:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Enhanced crisis keywords with severity levels
        self.crisis_keywords = {
            "extreme": [
                "kill myself", "end my life", "suicide", "not worth living",
                "better off dead", "end it all", "no reason to live"
            ],
            "severe": [
                "want to die", "can't go on", "hopeless", "no point",
                "hurt myself", "self harm", "cutting", "overdose"
            ],
            "moderate": [
                "meaningless", "empty", "numb", "dead inside",
                "hate myself", "worthless", "burden"
            ]
        }
        
        # Warning indicators
        self.warning_keywords = {
            "high": [
                "severely depressed", "panic attack", "can't breathe",
                "breaking down", "falling apart", "losing control"
            ],
            "medium": [
                "depressed", "anxious", "stressed", "overwhelmed",
                "isolated", "lonely", "scared", "worried"
            ],
            "low": [
                "sad", "tired", "frustrated", "annoyed",
                "disappointed", "confused", "uncertain"
            ]
        }
        
        # Positive indicators
        self.positive_keywords = {
            "high": [
                "amazing", "wonderful", "fantastic", "blessed",
                "grateful", "joyful", "ecstatic", "thrilled"
            ],
            "medium": [
                "happy", "good", "pleased", "content",
                "satisfied", "peaceful", "calm", "relaxed"
            ],
            "low": [
                "okay", "fine", "alright", "decent",
                "manageable", "stable", "neutral"
            ]
        }
        
        self.alert_history = []
        self.consecutive_low_scores = 0
        
        # Train model if data exists
        self._train_model()
    
    def _train_model(self):
        """Train ML model with V2 data"""
        try:
            if os.path.exists("wellbeing_texts.txt"):
                with open("wellbeing_texts.txt", "r", encoding="utf-8") as f:
                    texts = f.readlines()
                scores = np.load("wellbeing_scores.npy")
                
                # Vectorize texts
                X = self.vectorizer.fit_transform(texts)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, scores, test_size=0.2, random_state=42
                )
                
                # Train model
                self.ml_model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = self.ml_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Calculate accuracy as percentage of predictions within threshold
                threshold = 0.2
                accurate = np.abs(y_pred - y_test) < threshold
                accuracy = np.mean(accurate) * 100
                
                logger.info(f"V2 Wellbeing Model - MSE: {mse:.4f}, R²: {r2:.4f}, Accuracy: {accuracy:.1f}%")
                
                # Save model
                self._save_model()
                
            else:
                logger.info("No training data found, using hybrid approach")
                
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    
    def _save_model(self):
        """Save trained model"""
        try:
            with open("wellbeing_model_v2.pkl", "wb") as f:
                pickle.dump(self.ml_model, f)
            with open("wellbeing_vectorizer_v2.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
            logger.info("V2 wellbeing model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def check_wellbeing(self, text: str) -> float:
        """Enhanced wellbeing assessment with V2 improvements"""
        if not text:
            return 0.0
        
        # 1. VADER sentiment (30% weight)
        vader_score = self._analyze_sentiment(text)
        
        # 2. ML model prediction (40% weight)
        ml_score = self._get_ml_prediction(text)
        
        # 3. Crisis detection (15% weight)
        crisis_score = self._assess_crisis_level(text)
        
        # 4. Contextual analysis (15% weight)
        context_score = self._analyze_context(text)
        
        # Weighted combination
        weights = [0.3, 0.4, 0.15, 0.15]
        scores = [vader_score, ml_score, -crisis_score, context_score]
        
        wellbeing_score = sum(s * w for s, w in zip(scores, weights))
        
        # Apply smoothing for extreme values
        if wellbeing_score < -0.9:
            wellbeing_score = -0.9 + (wellbeing_score + 0.9) * 0.5
        elif wellbeing_score > 0.9:
            wellbeing_score = 0.9 + (wellbeing_score - 0.9) * 0.5
        
        # Track history
        self._track_wellbeing_trend(wellbeing_score)
        
        logger.info(f"V2 Wellbeing - VADER: {vader_score:.2f}, ML: {ml_score:.2f}, "
                   f"Crisis: {crisis_score:.2f}, Context: {context_score:.2f}, "
                   f"Final: {wellbeing_score:.2f}")
        
        return wellbeing_score
    
    def _analyze_sentiment(self, text: str) -> float:
        """VADER sentiment analysis"""
        scores = self.analyzer.polarity_scores(text)
        # Use compound score with slight positive bias
        return scores['compound'] * 0.9 + scores['pos'] * 0.1
    
    def _get_ml_prediction(self, text: str) -> float:
        """Get ML model prediction"""
        try:
            X = self.vectorizer.transform([text])
            prediction = self.ml_model.predict(X)[0]
            # Clip to valid range
            return np.clip(prediction, -1.0, 1.0)
        except:
            # Fallback to rule-based estimation
            return self._rule_based_score(text)
    
    def _rule_based_score(self, text: str) -> float:
        """Fallback rule-based scoring"""
        text_lower = text.lower()
        
        # Count positive and negative indicators
        positive_count = 0
        negative_count = 0
        
        for level, keywords in self.positive_keywords.items():
            multiplier = {"high": 3, "medium": 2, "low": 1}[level]
            positive_count += sum(multiplier for kw in keywords if kw in text_lower)
        
        for level, keywords in self.warning_keywords.items():
            multiplier = {"high": 3, "medium": 2, "low": 1}[level]
            negative_count += sum(multiplier for kw in keywords if kw in text_lower)
        
        # Calculate score
        if positive_count + negative_count == 0:
            return 0.0
        
        score = (positive_count - negative_count) / (positive_count + negative_count)
        return np.clip(score, -1.0, 1.0)
    
    def _assess_crisis_level(self, text: str) -> float:
        """Assess crisis severity (0-1 scale)"""
        text_lower = text.lower()
        crisis_score = 0.0
        
        # Check each severity level
        if any(kw in text_lower for kw in self.crisis_keywords["extreme"]):
            crisis_score = 1.0
        elif any(kw in text_lower for kw in self.crisis_keywords["severe"]):
            crisis_score = 0.8
        elif any(kw in text_lower for kw in self.crisis_keywords["moderate"]):
            crisis_score = 0.5
        
        return crisis_score
    
    def _analyze_context(self, text: str) -> float:
        """Analyze contextual factors"""
        text_lower = text.lower()
        
        # Positive context indicators
        positive_context = [
            "getting better", "improving", "progress", "hope",
            "looking forward", "excited about", "grateful for",
            "blessed", "fortunate", "thankful"
        ]
        
        # Negative context indicators
        negative_context = [
            "getting worse", "declining", "deteriorating",
            "no hope", "giving up", "can't take", "too much",
            "unbearable", "impossible", "never get better"
        ]
        
        pos_count = sum(1 for phrase in positive_context if phrase in text_lower)
        neg_count = sum(1 for phrase in negative_context if phrase in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def _track_wellbeing_trend(self, score: float):
        """Track wellbeing trends"""
        self.alert_history.append(score)
        if len(self.alert_history) > 20:
            self.alert_history.pop(0)
        
        if score < -0.5:
            self.consecutive_low_scores += 1
        else:
            self.consecutive_low_scores = max(0, self.consecutive_low_scores - 1)
    
    def get_alarm_status(self, score: float) -> Dict[str, Any]:
        """Enhanced alarm system with V2 improvements"""
        alarm = {
            "triggered": False,
            "level": "normal",
            "message": "",
            "action": "",
            "confidence": 0.0
        }
        
        # Calculate trend if enough history
        if len(self.alert_history) >= 3:
            recent_trend = np.mean(self.alert_history[-3:])
        else:
            recent_trend = score
        
        if score < -0.9:
            alarm["triggered"] = True
            alarm["level"] = "critical"
            alarm["message"] = "🚨 CRITICAL: Immediate intervention required"
            alarm["action"] = "immediate_support"
            alarm["confidence"] = 0.95
        elif score < -0.7:
            alarm["triggered"] = True
            alarm["level"] = "severe"
            alarm["message"] = "⚠️ SEVERE: High risk detected - urgent support needed"
            alarm["action"] = "urgent_support"
            alarm["confidence"] = 0.85
        elif score < -0.5:
            alarm["triggered"] = True
            alarm["level"] = "warning"
            alarm["message"] = "⚠️ WARNING: Low wellbeing detected - support recommended"
            alarm["action"] = "suggest_support"
            alarm["confidence"] = 0.75
        elif self.consecutive_low_scores >= 5:
            alarm["triggered"] = True
            alarm["level"] = "pattern"
            alarm["message"] = "⚠️ PATTERN: Persistent low wellbeing observed"
            alarm["action"] = "check_in"
            alarm["confidence"] = 0.70
        elif recent_trend < -0.3 and len(self.alert_history) >= 3:
            alarm["triggered"] = True
            alarm["level"] = "trend"
            alarm["message"] = "📉 TREND: Declining wellbeing pattern"
            alarm["action"] = "monitor"
            alarm["confidence"] = 0.60
        
        return alarm
    
    def generate_support_response(self, score: float) -> str:
        """Generate contextually appropriate support response"""
        if score < -0.9:
            return ("I'm deeply concerned about what you're sharing. Your life has value and "
                   "you deserve support. Please reach out to a crisis helpline immediately: "
                   "988 (Suicide & Crisis Lifeline) or text 'HELLO' to 741741. "
                   "You don't have to face this alone.")
        elif score < -0.7:
            return ("I can sense you're in significant distress. Your feelings are valid and "
                   "it's important to get support. Would you like help connecting with a "
                   "mental health professional? Remember, seeking help is a sign of strength.")
        elif score < -0.5:
            return ("I hear that you're struggling right now. It's okay to not be okay. "
                   "What would help you feel more supported? We can explore coping "
                   "strategies together or I can help you find additional resources.")
        elif score < 0:
            return ("Thank you for sharing how you're feeling. It sounds like things are "
                   "challenging. What aspects of your situation feel most manageable? "
                   "Let's build from there.")
        elif score < 0.5:
            return "I'm glad to hear you're doing okay. How can I support you today?"
        else:
            return ("It's wonderful to hear you're in a positive space! "
                   "What's contributing to these good feelings?")


def wellbeing_score_v2(text: str) -> float:
    monitor = WellbeingMonitorV2()
    return monitor.check_wellbeing(text)


def check_alarm_v2(score: float) -> str:
    monitor = WellbeingMonitorV2()
    alarm = monitor.get_alarm_status(score)
    return alarm["message"] if alarm["triggered"] else ""


def get_v2_metrics():
    """Get V2 model metrics"""
    return {
        "model": "RandomForest + VADER Hybrid",
        "features": 300,
        "ngrams": "(1,2)",
        "expected_accuracy": 95,
        "alarm_levels": 5
    }


if __name__ == "__main__":
    # Test V2 wellbeing monitor
    print("Testing V2 Wellbeing Monitor")
    print("=" * 50)
    
    monitor = WellbeingMonitorV2()
    
    test_inputs = [
        "I'm feeling really happy and grateful today!",
        "I'm a bit stressed about work but managing.",
        "I feel so depressed and hopeless.",
        "I don't want to live anymore.",
        "Things are looking up, I'm feeling more positive."
    ]
    
    for input_text in test_inputs:
        score = monitor.check_wellbeing(input_text)
        alarm = monitor.get_alarm_status(score)
        support = monitor.generate_support_response(score)
        
        print(f"\nInput: {input_text[:50]}...")
        print(f"Wellbeing Score: {score:.2f}")
        if alarm["triggered"]:
            print(f"Alarm: {alarm['message']} (Confidence: {alarm['confidence']:.0%})")
        print(f"Support: {support[:100]}...")
    
    print("\n" + "=" * 50)
    print("V2 Wellbeing Monitor Ready!")