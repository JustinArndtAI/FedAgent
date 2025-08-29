from .wellbeing_check import WellbeingMonitor, wellbeing_score, check_alarm

# Import V2 if available
try:
    from .wellbeing_check_v2 import WellbeingMonitorV2, wellbeing_score_v2, check_alarm_v2
    # Use V2 as default
    WellbeingMonitor = WellbeingMonitorV2
    wellbeing_score = wellbeing_score_v2
    check_alarm = check_alarm_v2
except ImportError:
    pass

__all__ = ['WellbeingMonitor', 'wellbeing_score', 'check_alarm']