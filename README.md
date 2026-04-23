📌 Overview

This project implements a self-pruning neural network using learnable gate parameters that automatically remove unnecessary weights during training.

The model is trained on the CIFAR-10 dataset and demonstrates the trade-off between:

🎯 Accuracy
✂️ Sparsity (model compression)

🚀 Key Features
Custom PrunableLinear layer with learnable gates
Automatic pruning using L1 regularization
Multi-layer neural network architecture
Evaluation across different sparsity strengths (λ values)
Visualization of:
Gate distributions
Training curves
Accuracy vs sparsity trade-off

🏗️ Model Architecture
Input (3×32×32 = 3072)
   ↓
Linear (3072 → 1024) + BN + ReLU
   ↓
Linear (1024 → 512) + BN + ReLU
   ↓
Linear (512 → 256) + BN + ReLU
   ↓
Linear (256 → 10)

✔ Only linear layers are prunable
✔ BatchNorm and ReLU remain unchanged

⚙️ How It Works

Each weight has an associated gate value:

effective_weight = weight × sigmoid(gate_score)
Gates close → weight becomes 0 (pruned)
L1 loss pushes gates toward zero → induces sparsity

🧪 Experiment Setup
Dataset: CIFAR-10
Optimizer: Adam
Scheduler: Cosine Annealing
Epochs: 30
Batch size: 256
λ (Sparsity Strengths)
[1e-4, 1e-3, 5e-3]
Low λ → High accuracy, low pruning
High λ → More pruning, possible accuracy drop
📦 Installation
pip install torch torchvision matplotlib numpy tqdm
▶️ Usage

Run the script:
python self_pruning_network.py

📊 Outputs
All outputs are saved in:
 outputs/gate_distributions.png
 outputs/training_curves.png
 outputs/summary_bar.png

📈 Results Interpretation
Bimodal gate distribution:
Many values near 0 → pruned weights
Others remain active
Increasing λ:
✅ Higher sparsity
❌ Potential accuracy drop

🧠 Key Concepts
Model pruning
Sparse neural networks
Regularization (L1)
Neural network compression

🔮 Future Improvements
Apply pruning to convolutional layers
Compare with magnitude-based pruning
Add inference speed benchmarking
Deploy compressed model
