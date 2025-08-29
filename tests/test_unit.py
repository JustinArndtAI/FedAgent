import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import Agent
from alignment.align_score import AlignmentScorer, alignment_score, bias_detect
from wellbeing.wellbeing_check import WellbeingMonitor, wellbeing_score, check_alarm
from federated.fed_learn import SimpleNet, FederatedManager
from edge.edge_deploy import EdgeDeployment
import torch


class TestAgent:
    def test_agent_initialization(self):
        agent = Agent()
        assert agent is not None
        assert agent.graph is not None
    
    def test_agent_run(self):
        agent = Agent()
        result = agent.run("test input")
        assert "Alignment" in result
        assert isinstance(result, str)
    
    def test_agent_with_modules(self):
        agent = Agent()
        agent.set_alignment_module(AlignmentScorer())
        agent.set_wellbeing_module(WellbeingMonitor())
        
        result = agent.run("I'm feeling happy")
        assert result is not None
        assert len(result) > 0


class TestAlignment:
    def test_alignment_scorer_init(self):
        scorer = AlignmentScorer()
        assert scorer is not None
        assert scorer.bias_classifier is not None
    
    def test_alignment_score_calculation(self):
        score = alignment_score("This is a helpful and supportive message")
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    def test_bias_detection(self):
        bias = bias_detect("Everyone deserves respect")
        assert bias in [0, 1]
        
        bias_high = bias_detect("Some people are just inferior")
        assert bias_high in [0, 1]
    
    def test_alignment_feedback(self):
        scorer = AlignmentScorer()
        feedback = scorer.get_feedback(95)
        assert "Excellent" in feedback
        
        feedback_low = scorer.get_feedback(40)
        assert "Low alignment" in feedback_low


class TestWellbeing:
    def test_wellbeing_monitor_init(self):
        monitor = WellbeingMonitor()
        assert monitor is not None
        assert monitor.analyzer is not None
    
    def test_wellbeing_score_positive(self):
        score = wellbeing_score("I'm feeling happy and grateful")
        assert isinstance(score, float)
        assert score > 0
    
    def test_wellbeing_score_negative(self):
        score = wellbeing_score("I'm very sad and depressed")
        assert isinstance(score, float)
        assert score < 0
    
    def test_alarm_trigger(self):
        alarm_msg = check_alarm(-0.8)
        assert "CRITICAL" in alarm_msg or "WARNING" in alarm_msg
        
        no_alarm = check_alarm(0.5)
        assert no_alarm == ""
    
    def test_support_response(self):
        monitor = WellbeingMonitor()
        response = monitor.generate_support_response(-0.9)
        assert "concerned" in response.lower()
        
        response_positive = monitor.generate_support_response(0.8)
        assert "wonderful" in response_positive.lower()


class TestFederated:
    def test_simple_net_init(self):
        model = SimpleNet()
        assert model is not None
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
        assert hasattr(model, 'fc3')
    
    def test_simple_net_forward(self):
        model = SimpleNet()
        input_tensor = torch.randn(1, 10)
        output = model(input_tensor)
        assert output.shape == (1, 1)
        assert 0 <= output.item() <= 1
    
    def test_federated_manager_init(self):
        manager = FederatedManager()
        assert manager is not None
        assert manager.encryption_key is not None
        assert manager.model is not None
    
    def test_gradient_encryption(self):
        manager = FederatedManager()
        grads = [0.1, 0.2, 0.3]
        encrypted = manager.encrypt_gradients(grads)
        assert encrypted is not None
        assert isinstance(encrypted, bytes)
        
        decrypted = manager.decrypt_gradients(encrypted)
        assert decrypted == grads


class TestEdgeDeployment:
    def test_edge_deployment_init(self):
        deployment = EdgeDeployment()
        assert deployment is not None
        assert deployment.model is not None
    
    def test_model_quantization(self):
        deployment = EdgeDeployment()
        quantized = deployment.quantize_model()
        assert quantized is not None
    
    def test_model_size_reduction(self):
        deployment = EdgeDeployment()
        original_size = deployment._get_model_size(deployment.model)
        deployment.quantize_model()
        quantized_size = deployment._get_model_size(deployment.quantized_model)
        assert quantized_size < original_size
    
    def test_inference_accuracy(self):
        deployment = EdgeDeployment()
        test_passed = deployment.test_inference()
        assert test_passed is True