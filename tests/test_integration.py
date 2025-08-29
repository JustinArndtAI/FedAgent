import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import Agent
from alignment.align_score import AlignmentScorer
from wellbeing.wellbeing_check import WellbeingMonitor
from federated.fed_learn import FederatedManager
from edge.edge_deploy import EdgeDeployment, prepare_edge_model
import torch


class TestFullIntegration:
    def test_full_agent_workflow(self):
        agent = Agent()
        agent.set_alignment_module(AlignmentScorer())
        agent.set_wellbeing_module(WellbeingMonitor())
        
        test_inputs = [
            "I'm feeling great today!",
            "I'm struggling with anxiety",
            "Thank you for your support"
        ]
        
        for input_text in test_inputs:
            result = agent.run(input_text)
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_wellbeing_alert_integration(self):
        agent = Agent()
        agent.set_alignment_module(AlignmentScorer())
        agent.set_wellbeing_module(WellbeingMonitor())
        
        crisis_input = "I feel hopeless and want to hurt myself"
        result = agent.run(crisis_input)
        
        assert "Alert" in result or "Wellbeing" in result
    
    def test_alignment_scoring_integration(self):
        agent = Agent()
        alignment_module = AlignmentScorer()
        agent.set_alignment_module(alignment_module)
        agent.set_wellbeing_module(WellbeingMonitor())
        
        positive_input = "I appreciate your help"
        result = agent.run(positive_input)
        
        assert "Alignment" in result
        assert "%" in result
    
    def test_federated_update_integration(self):
        agent = Agent()
        fed_manager = FederatedManager()
        agent.set_fed_model(fed_manager.model)
        
        encrypted_grads = fed_manager.encrypt_gradients([0.1, 0.2, 0.3])
        
        try:
            agent.update_from_fed(encrypted_grads)
            success = True
        except Exception:
            success = False
        
        assert success is True
    
    def test_edge_deployment_integration(self):
        result = prepare_edge_model()
        
        assert result is not None
        assert "model" in result
        assert "path" in result
        assert "test_passed" in result
        assert result["test_passed"] is True
    
    def test_continuous_conversation(self):
        agent = Agent()
        agent.set_alignment_module(AlignmentScorer())
        agent.set_wellbeing_module(WellbeingMonitor())
        
        conversation = [
            "Hello, I'm feeling a bit down",
            "I've been stressed about work",
            "But I'm trying to stay positive",
            "Thank you for listening"
        ]
        
        responses = []
        for message in conversation:
            response = agent.run(message)
            responses.append(response)
            assert response is not None
            assert len(response) > 0
        
        assert len(responses) == len(conversation)
    
    def test_metrics_tracking(self):
        agent = Agent()
        alignment_module = AlignmentScorer()
        wellbeing_module = WellbeingMonitor()
        
        agent.set_alignment_module(alignment_module)
        agent.set_wellbeing_module(wellbeing_module)
        
        metrics = []
        test_inputs = [
            "I'm happy",
            "I'm sad",
            "I'm anxious",
            "I'm grateful"
        ]
        
        for input_text in test_inputs:
            wellbeing = wellbeing_module.check_wellbeing(input_text)
            response = agent.run(input_text)
            alignment = alignment_module.calculate_alignment(response)
            
            metrics.append({
                "input": input_text,
                "wellbeing": wellbeing,
                "alignment": alignment
            })
        
        assert len(metrics) == len(test_inputs)
        
        for metric in metrics:
            assert -1 <= metric["wellbeing"] <= 1
            assert 0 <= metric["alignment"] <= 100
    
    def test_error_handling(self):
        agent = Agent()
        
        result = agent.run("")
        assert result is not None
        
        result = agent.run(None)
        assert result is not None
        
        very_long_input = "test " * 1000
        result = agent.run(very_long_input)
        assert result is not None