from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
from typing import Dict, Any, List
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WellbeingMonitor:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.crisis_keywords = [
            "suicide", "kill myself", "end it all", "no point", "hopeless",
            "can't go on", "want to die", "better off dead", "self harm",
            "hurt myself", "cutting", "overdose"
        ]
        self.warning_keywords = [
            "depressed", "anxious", "panic", "scared", "lonely",
            "isolated", "worthless", "failure", "hate myself",
            "can't cope", "breaking down", "falling apart"
        ]
        self.positive_keywords = [
            "happy", "joy", "excited", "grateful", "blessed",
            "wonderful", "amazing", "great", "fantastic", "love",
            "peaceful", "calm", "content", "satisfied"
        ]
        self.alert_history = []
        self.consecutive_low_scores = 0
    
    def check_wellbeing(self, text: str) -> float:
        if not text:
            return 0.0
        
        sentiment_score = self._analyze_sentiment(text)
        
        crisis_level = self._check_crisis_indicators(text)
        
        warning_level = self._check_warning_signs(text)
        
        positive_level = self._check_positive_indicators(text)
        
        wellbeing_score = self._calculate_composite_score(
            sentiment_score, crisis_level, warning_level, positive_level
        )
        
        self._track_wellbeing_trend(wellbeing_score)
        
        logger.info(f"Wellbeing assessment - Score: {wellbeing_score:.2f}, "
                   f"Sentiment: {sentiment_score:.2f}, "
                   f"Crisis: {crisis_level}, Warning: {warning_level}")
        
        return wellbeing_score
    
    def _analyze_sentiment(self, text: str) -> float:
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']
    
    def _check_crisis_indicators(self, text: str) -> int:
        text_lower = text.lower()
        crisis_count = sum(1 for keyword in self.crisis_keywords if keyword in text_lower)
        return min(3, crisis_count)
    
    def _check_warning_signs(self, text: str) -> int:
        text_lower = text.lower()
        warning_count = sum(1 for keyword in self.warning_keywords if keyword in text_lower)
        return min(5, warning_count)
    
    def _check_positive_indicators(self, text: str) -> int:
        text_lower = text.lower()
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        return min(5, positive_count)
    
    def _calculate_composite_score(self, sentiment: float, crisis: int, 
                                  warning: int, positive: int) -> float:
        base_score = sentiment
        
        base_score -= (crisis * 0.5)
        
        base_score -= (warning * 0.1)
        
        base_score += (positive * 0.1)
        
        return max(-1.0, min(1.0, base_score))
    
    def _track_wellbeing_trend(self, score: float):
        self.alert_history.append(score)
        if len(self.alert_history) > 10:
            self.alert_history.pop(0)
        
        if score < -0.5:
            self.consecutive_low_scores += 1
        else:
            self.consecutive_low_scores = 0
    
    def get_alarm_status(self, score: float) -> Dict[str, Any]:
        alarm = {
            "triggered": False,
            "level": "normal",
            "message": "",
            "action": ""
        }
        
        if score < -0.8:
            alarm["triggered"] = True
            alarm["level"] = "critical"
            alarm["message"] = "⚠️ CRITICAL: Immediate wellbeing intervention needed"
            alarm["action"] = "pause_and_support"
        elif score < -0.5:
            alarm["triggered"] = True
            alarm["level"] = "warning"
            alarm["message"] = "⚠️ WARNING: Low wellbeing detected - please take a break"
            alarm["action"] = "suggest_break"
        elif self.consecutive_low_scores >= 3:
            alarm["triggered"] = True
            alarm["level"] = "concern"
            alarm["message"] = "⚠️ PATTERN: Consistent low wellbeing observed"
            alarm["action"] = "check_in"
        
        return alarm
    
    def generate_support_response(self, score: float) -> str:
        if score < -0.8:
            return ("I'm really concerned about what you're sharing. Your wellbeing is important. "
                   "Please consider reaching out to a mental health professional or crisis helpline. "
                   "You don't have to go through this alone.")
        elif score < -0.5:
            return ("I can sense you're going through a difficult time. It's okay to feel this way. "
                   "Would you like to take a break or talk about what's on your mind?")
        elif score < 0:
            return ("I hear that things might be challenging right now. Remember that it's okay "
                   "to have difficult days. What would help you feel better?")
        elif score < 0.5:
            return "Thank you for sharing. How can I best support you today?"
        else:
            return "It's wonderful to see you in good spirits! Keep up the positive energy!"


def wellbeing_score(text: str) -> float:
    monitor = WellbeingMonitor()
    return monitor.check_wellbeing(text)


def check_alarm(score: float) -> str:
    monitor = WellbeingMonitor()
    alarm = monitor.get_alarm_status(score)
    return alarm["message"] if alarm["triggered"] else ""