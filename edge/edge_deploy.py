import torch
import torch.quantization as quantization
from federated.fed_learn import SimpleNet
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EdgeDeployment:
    def __init__(self):
        self.model = SimpleNet()
        self.quantized_model = None
        
    def quantize_model(self):
        logger.info("Starting model quantization for edge deployment...")
        
        self.model.eval()
        
        self.quantized_model = quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        original_size = self._get_model_size(self.model)
        quantized_size = self._get_model_size(self.quantized_model)
        
        compression_ratio = (1 - quantized_size / original_size) * 100
        
        logger.info(f"Original model size: {original_size / 1024:.2f} KB")
        logger.info(f"Quantized model size: {quantized_size / 1024:.2f} KB")
        logger.info(f"Compression ratio: {compression_ratio:.1f}%")
        
        return self.quantized_model
    
    def _get_model_size(self, model):
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return param_size + buffer_size
    
    def save_for_mobile(self, path="edge_model.pt"):
        if self.quantized_model is None:
            self.quantize_model()
        
        example_input = torch.randn(1, 10)
        traced_model = torch.jit.trace(self.quantized_model, example_input)
        
        traced_model.save(path)
        logger.info(f"Model saved for mobile deployment at: {path}")
        
        return path
    
    def test_inference(self):
        if self.quantized_model is None:
            self.quantize_model()
        
        test_input = torch.randn(5, 10)
        
        self.model.eval()
        with torch.no_grad():
            original_output = self.model(test_input)
        
        self.quantized_model.eval()
        with torch.no_grad():
            quantized_output = self.quantized_model(test_input)
        
        mse = torch.mean((original_output - quantized_output) ** 2)
        logger.info(f"Inference test MSE: {mse.item():.6f}")
        
        return mse.item() < 0.01


def prepare_edge_model():
    deployment = EdgeDeployment()
    
    quantized_model = deployment.quantize_model()
    
    model_path = deployment.save_for_mobile()
    
    test_passed = deployment.test_inference()
    
    return {
        "model": quantized_model,
        "path": model_path,
        "test_passed": test_passed
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Edge Deployment - Model Quantization")
    print("=" * 60)
    
    result = prepare_edge_model()
    
    if result["test_passed"]:
        print("✓ Model quantization successful!")
        print(f"✓ Model saved at: {result['path']}")
        print("✓ Inference test passed")
    else:
        print("✗ Model quantization failed inference test")
    
    print("\nModel is ready for edge deployment!")