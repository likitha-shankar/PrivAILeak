# PrivAI-Leak: Privacy Auditing Framework for LLMs

> **A comprehensive framework for detecting and mitigating information leakage in large language models through Differential Privacy**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

PrivAI-Leak is a privacy-auditing framework that detects and mitigates information leakage in large language models by applying **Differentially Private Stochastic Gradient Descent (DP-SGD)** during training. This project demonstrates the privacy risks of LLMs and provides practical solutions through differential privacy mechanisms.

### 🏥 Healthcare Focus

**Primary Use Case:** Healthcare AI systems that analyze patient records (PHI - Protected Health Information)

- Hospitals deploying AI assistants for doctors
- Patient data privacy protection (HIPAA compliance)
- Medical record analysis with privacy guarantees
- Synthetic healthcare data generation with embedded PII

### Key Features

- 🔍 **Privacy Attack Simulation**: Membership inference and prompt extraction attacks
- 🔒 **Differential Privacy Training**: Manual DP-SGD implementation for text generation models
- 📊 **Comprehensive Evaluation**: Privacy-utility trade-off analysis across multiple epsilon values
- 📈 **Visualization**: Detailed plots and comparison charts
- 🧪 **Synthetic Dataset**: Realistic PII-embedded healthcare text generation
- 🎓 **Research-Ready**: Complete framework for privacy auditing experiments

---

## ⚡ Quick Start

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+
- CUDA-capable GPU (recommended for faster training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/likitha-shankar/PrivAILeak.git
cd PrivAILeak
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the complete pipeline**
```bash
python main.py
```

The pipeline will:
1. Generate synthetic healthcare dataset with embedded PII
2. Train a baseline model (GPT-2)
3. Run privacy attacks on the baseline model
4. Train differentially private models with various epsilon values
5. Evaluate and compare all models
6. Generate visualizations

---

## 📁 Project Structure

```
PrivAILeak/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── config.py                          # Configuration parameters
├── main.py                            # Main pipeline orchestrator
├── requirements.txt                   # Python dependencies
├── FINAL_REPORT_COMPREHENSIVE.md      # Comprehensive project report
│
├── src/                               # Source code modules
│   ├── __init__.py
│   ├── healthcare_data_generator.py   # Synthetic healthcare data generation
│   ├── baseline_training.py           # Baseline model training (GPT-2)
│   ├── privacy_attacks.py             # Privacy attack implementations
│   ├── dp_training_manual.py          # Manual DP-SGD training
│   ├── advanced_privacy_attacks_enhanced.py  # Enhanced attack methods
│   ├── evaluation.py                  # Model evaluation metrics
│   └── visualization.py               # Results visualization
│
├── data/                              # Generated datasets
├── models/                            # Trained model checkpoints
│   ├── baseline_model/                # Baseline GPT-2 model
│   └── dp_model_eps_*/                # DP models for each epsilon
├── results/                           # Evaluation results and metrics
└── logs/                              # Training logs
```

---

## 🛠️ Configuration

Edit `config.py` to customize training and evaluation parameters:

```python
# Model settings
MODEL_NAME = "gpt2"                    # Base model (GPT-2)
NUM_EPOCHS = 5                         # Training epochs
BATCH_SIZE = 4                      # Batch size
LEARNING_RATE = 3e-5                   # Learning rate

# Differential Privacy parameters
EPSILON_VALUES = [0.5, 1.0, 5.0, 10.0]  # Privacy budgets to test
DELTA = 1e-5                           # DP delta parameter
MAX_GRAD_NORM = 1.0                    # Gradient clipping threshold

# Dataset configuration
NUM_TRAIN_SAMPLES = 2000               # Training samples
NUM_TEST_SAMPLES = 400                 # Test samples
NUM_PRIVATE_RECORDS = 150              # Records with PII
PRIVATE_RATIO = 0.15                   # Ratio of data containing PHI
```

---

## 📊 Expected Results

| Model | Privacy Budget (ε) | PII Leakage | Perplexity | Quality |
|-------|-------------------|-------------|------------|---------|
| Baseline | ∞ (No privacy) | ~40% | ~24.5 | ⭐⭐⭐⭐⭐ |
| DP Model | 0.5 | ~15% | ~32.1 | ⭐⭐⭐ |
| DP Model | 1.0 (Recommended) | ~21% | ~27.9 | ⭐⭐⭐⭐ |
| DP Model | 5.0 | ~28% | ~26.2 | ⭐⭐⭐⭐ |
| DP Model | 10.0 | ~35% | ~25.1 | ⭐⭐⭐⭐⭐ |

**Key Finding:** DP-SGD with ε=1.0 reduces privacy leakage by ~47% with only ~14% quality degradation, providing an excellent privacy-utility trade-off.

---

## 🚀 Usage Examples

### Run Complete Pipeline
```bash
python main.py
```

### Run Individual Components

```bash
# Generate synthetic healthcare data
python -m src.healthcare_data_generator

# Train baseline model
python -m src.baseline_training

# Run privacy attacks
python -m src.privacy_attacks

# Train DP models
python -m src.dp_training_manual

# Evaluate models
python -m src.evaluation

# Generate visualizations
python -m src.visualization
```

### Demo Scripts

```bash
# Quick demo
python quick_demo.py

# Presentation demo
python presentation_demo.py

# Test with your own data
python test_your_own_data.py
```

---

## 🔬 Research Contributions

### What This Project Demonstrates

1. **Privacy Risk**: LLMs memorize and leak PII from training data (demonstrated with ~40% baseline leakage)
2. **DP Solution**: DP-SGD effectively mitigates leakage (reduces to ~21% with ε=1.0)
3. **Trade-off Analysis**: Privacy comes at acceptable quality cost (~14% perplexity increase)
4. **Practical Applicability**: Framework applicable to real-world sensitive domains (healthcare, legal, finance)

### Key Contributions

- Comprehensive privacy-utility trade-off analysis across multiple epsilon values
- Manual DP-SGD implementation optimized for text generation models
- Multi-metric evaluation (perplexity, BLEU, ROUGE, diversity)
- Demonstration that DP-SGD is practical for real-world deployment
- Healthcare-focused synthetic data generation with realistic PHI

### Research Gap Addressed

While Differential Privacy has been widely applied to numerical and tabular data, **very few studies have explored its effect on text-generation models**. Existing LLMs lack formal privacy guarantees and remain vulnerable to data-memorization attacks.

**PrivAI-Leak addresses this gap** by providing:
- Experimental framework to quantify privacy leakage in language models
- Practical DP-based training implementation for text generation
- Comprehensive evaluation methodology for privacy-utility trade-offs

---

## ⚠️ Limitations

1. **Utility Loss**: Adding DP noise reduces text-generation quality and accuracy
2. **Scalability**: DP-SGD slows training and requires smaller batch sizes
3. **Synthetic Data**: Uses fake PII; results approximate but don't fully model real sensitivity
4. **Partial Defense**: DP protects training data but not post-training attacks (prompt injection, jailbreaks)
5. **Computational Cost**: Privacy-preserving training is 2-3x slower than standard training

---

## 📚 Technical Details

### Architecture

- **Base Model**: GPT-2 (124M parameters) from Hugging Face
- **DP Implementation**: Manual DP-SGD with gradient clipping and noise addition
- **Training**: PyTorch with custom training loops
- **Evaluation**: Multiple metrics including perplexity, BLEU, ROUGE, and diversity scores

### Privacy Attacks Implemented

1. **Membership Inference Attacks**: Determine if specific data was in training set
2. **Prompt Extraction Attacks**: Extract PII through carefully crafted prompts
3. **Enhanced Attacks**: Advanced techniques for better attack success rates

---

## 📚 References

1. **Differential Privacy**: [Dwork, C. (2006). Differential Privacy](https://link.springer.com/chapter/10.1007/11787006_1)
2. **DP-SGD**: [Abadi et al. (2016). Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)
3. **LLM Privacy Risks**: [Carlini et al. (2021). Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805)
4. **GPT-2**: [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Hugging Face** - Pre-trained GPT-2 model and Transformers library
- **PyTorch Team** - Deep learning framework
- **OpenAI** - GPT-2 architecture and research
- **Research Community** - Differential privacy and LLM privacy research

---

## 🎓 Academic Use

If you use this framework in your research, please cite:

```bibtex
@software{privaileak2024,
  title={PrivAI-Leak: Privacy Auditing Framework for Large Language Models},
  author={Likitha Shankar},
  year={2024},
  url={https://github.com/likitha-shankar/PrivAILeak}
}
```

---

## 📧 Contact

For questions or issues, please open an issue on [GitHub](https://github.com/likitha-shankar/PrivAILeak/issues).

---

**⭐ If you find this project helpful, please consider giving it a star!**

PrivAI-Leak innovates by transforming differential privacy from a static data-protection mechanism into a dynamic, model-level defense against information leakage in large language models — addressing a critical gap in current privacy-preserving LLM research.

---

*Made for DPS Masters Course Project* 🎓
