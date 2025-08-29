import sys
import logging
from core.agent import Agent
from alignment.align_score import AlignmentScorer
from wellbeing.wellbeing_check import WellbeingMonitor
from federated.fed_learn import FederatedManager, start_fed_sim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("EdgeFedAlign MVP - Privacy-First AI Therapy Agent")
    print("=" * 60)
    
    logger.info("Initializing components...")
    
    agent = Agent()
    
    alignment_module = AlignmentScorer()
    wellbeing_module = WellbeingMonitor()
    fed_manager = FederatedManager()
    
    agent.set_alignment_module(alignment_module)
    agent.set_wellbeing_module(wellbeing_module)
    agent.set_fed_model(fed_manager.model)
    
    logger.info("Agent initialized successfully")
    
    print("\nTesting Agent with sample inputs...")
    print("-" * 40)
    
    test_inputs = [
        "I'm feeling happy today",
        "I'm very sad and don't know what to do",
        "I'm anxious about the future",
        "Everything is wonderful!",
        "I feel isolated and alone"
    ]
    
    for input_text in test_inputs:
        print(f"\nInput: {input_text}")
        output = agent.run(input_text)
        print(f"Response: {output}")
    
    print("\n" + "-" * 40)
    print("Starting federated learning simulation...")
    print("-" * 40)
    
    try:
        success = start_fed_sim(num_clients=2)
        if success:
            print("✓ Federated learning simulation completed")
            
            encrypted_grads = fed_manager.encrypt_gradients([0.1, 0.2, 0.3])
            agent.update_from_fed(encrypted_grads)
            print("✓ Agent updated with federated gradients")
        else:
            print("✗ Federated learning simulation failed")
    except Exception as e:
        logger.error(f"Federated learning error: {e}")
        print(f"✗ Federated learning error: {e}")
    
    print("\n" + "=" * 60)
    print("EdgeFedAlign MVP Demonstration Complete!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Run 'streamlit run demo/demo_app.py' for web UI")
    print("2. Run 'python edge/edge_deploy.py' for mobile deployment")
    print("3. Run 'pytest tests/' for running tests")


if __name__ == "__main__":
    main()