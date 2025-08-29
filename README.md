# EdgeFedAlign - Privacy-First AI Therapy Agent 🚀

**V3 ULTIMATE ACCURACY MODE - BEAST MODE ACTIVATED**

A Python framework for deploying autonomous AI agents that run on edge devices, learn collaboratively via federation without sharing raw data, echo alignment scores in real-time, and raise wellbeing alarms.

## 📊 V3 Performance Metrics (Latest)

| Metric | V1 | V2 | **V3** | Target | Status |
|--------|----|----|--------|--------|--------|
| **Alignment Accuracy** | 61.0% | 66.5% | **87.3%** | 95% | 🔥 Beast Mode |
| **Wellbeing Detection** | 66.7% | 83.3% | **87.1%** | 98% | 🔥 Beast Mode |
| **Response Time** | 10ms | 767ms | **16.3ms** | <50ms | ✅ Achieved |
| **Model R²** | - | 0.83 | **0.96** | >0.95 | ✅ Achieved |

### V3 Highlights
- **XGBoost** alignment scoring with 2000 features & trigrams
- **BERT embeddings** (all-MiniLM-L6-v2) for wellbeing
- **Ensemble models**: GradientBoosting + RandomForest
- **10,000 training samples** generated for ultimate accuracy
- **97.9% faster** than V2 while maintaining higher accuracy

## Features

- **Federated Learning**: Collaborative learning without sharing raw user data
- **Edge Deployment**: Optimized for mobile and edge devices with model quantization
- **Alignment Monitoring**: Real-time ethical alignment scoring
- **Wellbeing Alerts**: Automatic detection and intervention for mental health concerns
- **Privacy-First**: Zero user data storage, encrypted gradient sharing
- **Multi-Platform**: Web UI (Streamlit), Mobile (Kivy), CLI support

## V3 Files & Documentation

### Key V3 Files
- `alignment/align_score_v3.py` - XGBoost alignment scorer
- `wellbeing/wellbeing_check_v3.py` - BERT/ensemble wellbeing monitor  
- `main_v3.py` - V3 agent with ultimate accuracy
- `run_v3_tests.py` - Comprehensive V3 test suite
- `V3_ULTIMATE_RESULTS.md` - Detailed V3 performance report
- `v3_performance_metrics.png` - Performance visualizations
- Model files: `align_xgb_v3_model.json`, `wellbeing_primary_v3.pkl`, etc.

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/JustinArndtAI/FedAgent.git
cd FedAgent
```

2. Create and activate virtual environment:
```bash
python -m venv edgefedalign_env

# Windows
edgefedalign_env\Scripts\activate

# Mac/Linux
source edgefedalign_env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

#### V3 Ultimate Accuracy Mode (Recommended)
```bash
python main_v3.py --demo
```

#### Command Line Demo (V1)
```bash
python main.py
```

#### Web UI (Streamlit)
```bash
streamlit run demo/demo_app.py
```

#### Mobile App (Kivy)
```bash
python edge/mobile_app.py
```

#### Edge Deployment
```bash
python edge/edge_deploy.py
```

### Running Tests

#### V3 Comprehensive Tests
```bash
python run_v3_tests.py
```

#### Standard Test Suite
```bash
pytest tests/ -v
```

## Project Structure

```
EdgeFedAlign/
├── core/               # Core agent implementation
│   ├── agent.py       # Main agent class with LangGraph
│   └── __init__.py
├── federated/          # Federated learning components
│   ├── fed_learn.py   # Flower-based federated learning
│   └── __init__.py
├── alignment/          # Alignment scoring system
│   ├── align_score.py # V1: Bias detection and alignment metrics
│   ├── align_score_v2.py # V2: ML-enhanced with RandomForest
│   ├── align_score_v3.py # V3: XGBoost with 2000 features
│   └── __init__.py
├── wellbeing/          # Mental health monitoring
│   ├── wellbeing_check.py # V1: VADER sentiment analysis
│   ├── wellbeing_check_v2.py # V2: TF-IDF + LogisticRegression
│   ├── wellbeing_check_v3.py # V3: BERT + Ensemble models
│   └── __init__.py
├── edge/               # Edge deployment tools
│   ├── edge_deploy.py # Model quantization
│   ├── mobile_app.py  # Kivy mobile app
│   └── __init__.py
├── demo/               # Demo applications
│   ├── demo_app.py    # Streamlit web UI
│   └── __init__.py
├── tests/              # Test suite
│   ├── test_unit.py
│   ├── test_integration.py
│   └── __init__.py
├── main.py             # V1 entry point
├── main_v3.py          # V3 entry point (BEAST MODE)
├── run_v3_tests.py     # V3 comprehensive tests
├── data_gen_v3.py      # V3 dataset generator (10k samples)
├── V3_ULTIMATE_RESULTS.md # V3 performance report
├── requirements.txt    # Dependencies
└── README.md          # Documentation
```

## Key Components

### 1. Agent (LangGraph-based)
- Therapy response generation
- Workflow orchestration
- Module integration

### 2. Federated Learning (Flower)
- Encrypted gradient sharing
- Privacy-preserving model updates
- Multi-client simulation

### 3. Alignment Scoring
- Bias detection
- Helpfulness assessment
- Harmfulness prevention
- Coherence evaluation

### 4. Wellbeing Monitoring
- Sentiment analysis (VADER)
- Crisis keyword detection
- Alert generation
- Support response generation

### 5. Edge Deployment
- PyTorch model quantization
- Mobile optimization
- Resource-constrained inference

## Privacy & Security

- **No Data Storage**: All processing happens locally
- **Encrypted Communications**: Gradients encrypted with Fernet
- **On-Device Processing**: Models run directly on user devices
- **Federated Updates**: Learn from all users without seeing their data
- **GDPR/HIPAA Compliant Design**: Privacy-first architecture

## API Usage

### Basic Agent Usage
```python
from core.agent import Agent
from alignment.align_score import AlignmentScorer
from wellbeing.wellbeing_check import WellbeingMonitor

# Initialize agent
agent = Agent()
agent.set_alignment_module(AlignmentScorer())
agent.set_wellbeing_module(WellbeingMonitor())

# Process input
response = agent.run("I'm feeling anxious today")
print(response)
```

### Federated Learning
```python
from federated.fed_learn import start_fed_sim

# Run federated simulation
success = start_fed_sim(num_clients=3)
```

### Edge Deployment
```python
from edge.edge_deploy import prepare_edge_model

# Quantize and save model
result = prepare_edge_model()
print(f"Model saved at: {result['path']}")
```

## Testing

Run the full test suite:
```bash
pytest tests/ -v
```

Run specific test categories:
```bash
# Unit tests only
pytest tests/test_unit.py -v

# Integration tests only
pytest tests/test_integration.py -v
```

## Performance Metrics

### V3 (Current - BEAST MODE)
- **Alignment Accuracy**: 87.3% (XGBoost, 2000 features, trigrams)
- **Wellbeing Detection**: 87.1% (BERT + Ensemble)
- **Response Time**: 16.3ms average
- **Model R²**: 0.96 (both models)
- **Training Data**: 10,000 realistic samples

### V2
- **Alignment Accuracy**: 66.5% (RandomForest)
- **Wellbeing Detection**: 83.3% (TF-IDF + LogisticRegression)
- **Response Time**: 767ms average

### V1 (Baseline)
- **Alignment Accuracy**: 61.0% (rule-based)
- **Wellbeing Detection**: 66.7% (VADER only)
- **Response Time**: 10ms average
- **Model Size Reduction**: ~60-70% via quantization
- **Federation Efficiency**: 3-client simulation in <30s

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- LangChain/LangGraph for agent orchestration
- Flower for federated learning
- VADER for sentiment analysis
- PyTorch for model quantization
- Streamlit for web UI
- Kivy for mobile development

## Contact

Project Link: [https://github.com/JustinArndtAI/FedAgent](https://github.com/JustinArndtAI/FedAgent)

## Next Steps

- [ ] Implement real device federation
- [ ] Add more sophisticated alignment metrics
- [ ] Enhance crisis intervention protocols
- [ ] Build production mobile apps
- [ ] Add multi-language support
- [ ] Implement differential privacy
- [ ] Create admin dashboard
- [ ] Add voice interface support