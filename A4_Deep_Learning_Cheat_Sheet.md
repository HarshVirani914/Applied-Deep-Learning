# 📝 Applied Deep Learning - A4 Exam Cheat Sheet

## 🧠 **1. NEURAL NETWORK FUNDAMENTALS**

### **Basic Architecture**

- **Forward Pass**: `y = σ(Wx + b)`
- **Loss Function**: `L = 1/n Σ(y_true - y_pred)²` (MSE) or `CrossEntropy`
- **Backpropagation**: `∂L/∂w = x(y_pred - y_true)`, `∂L/∂b = (y_pred - y_true)`
- **Parameter Update**: `w = w - α∇w`, `b = b - α∇b`

### **ANN Visual Architecture**

```
Input Layer    Hidden Layer    Output Layer
   x₁ ────────── h₁ ──────────── y₁
   x₂ ──┬─────── h₂ ──┬───────── y₂
   x₃ ──┴─────── h₃ ──┴───────── y₃
     (Dense connections)   (Dense connections)
```

### **Activation Functions**

- **ReLU**: `f(x) = max(0,x)` - Most common, solves vanishing gradients
- **Sigmoid**: `f(x) = 1/(1+e^-x)` - Output [0,1], saturates
- **Tanh**: `f(x) = (e^x-e^-x)/(e^x+e^-x)` - Output [-1,1], zero-centered
- **Softmax**: `f(x_i) = e^x_i/Σe^x_j` - Multi-class classification

### **Batch Normalization**

- **Formula**: `BN(x) = γ((x-μ)/σ) + β`
- **Benefits**: Faster training, higher learning rates, regularization
- **When**: Usually before activation function
- **Why Works**: Smooths optimization landscape, reduces layer interdependence

## 🏗️ **2. CNN ARCHITECTURE**

### **CNN Visual Architecture**

```
Input Image → [Conv2d] → [BatchNorm] → [ReLU] → [MaxPool] → [Conv2d] → ... → [Flatten] → [Linear] → Output
   28×28         5×5 filters    BN         ReLU      2×2        3×3                                   10 classes
             ↓                              ↓         ↓
         Feature Maps              Activation    Downsampling
```

### **Convolution Operation Visual**

```
Input:     Filter:    Output:
1 2 3      1 0        (1×1)+(2×0)+(4×1)+(5×0) = 5
4 5 6   *  0 1    =   (2×1)+(3×0)+(5×1)+(6×0) = 7
7 8 9                 (4×1)+(5×0)+(7×1)+(8×0) = 11
```

### **Core Operations**

- **Convolution**: `(f*g)[n] = Σf[m]g[n-m]` (actually cross-correlation)
- **Padding**: 'SAME' keeps size, 'VALID' reduces size
- **Stride**: Step size for filter movement
- **Pooling**: MaxPool takes maximum, AvgPool takes average

### **Why CNNs Excel at Images**

1. **Weight Sharing**: Same filter applied everywhere → translation invariance
2. **Local Connectivity**: Each neuron sees small spatial region
3. **Hierarchical Features**: Edges → Shapes → Objects
4. **Parameter Efficiency**: ~10x fewer parameters than FC
5. **Spatial Preservation**: Maintains image structure

### **Classic Architectures**

- **LeNet-5** (1998): Conv→Pool→Conv→Pool→FC (60K params)
- **AlexNet** (2012): ReLU, Dropout, 60M params
- **VGG** (2014): Deep with 3×3 filters only
- **ResNet** (2015): Skip connections solve vanishing gradients

### **Dimensions**

- **Input**: `(B, C, H, W)`
- **Conv2d**: `out = (in + 2p - k)/s + 1`
- **MaxPool**: Reduces spatial dimensions

## 🔄 **3. RNN ARCHITECTURES**

### **Vanilla RNN Visual**

```
Input:  x₁ → x₂ → x₃ → x₄
         ↓    ↓    ↓    ↓
State:  h₁ → h₂ → h₃ → h₄
         ↓    ↓    ↓    ↓
Output: y₁   y₂   y₃   y₄

h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
```

### **LSTM Architecture**

```
      ┌─────────────── C_{t-1} ─────────────→ C_t ──┐
      │                  ↓                    ↑     │
   ┌──▼──┐         ┌────▼────┐         ┌────▲────┐ │
h_{t-1}│ f_t │ ×   │  tanh   │    ×    │   i_t   │ │
   └─────┘         └─────────┘         └─────────┘ │
      ↑                  ↑                    ↑     │
   ┌──▼────────────────▼────────────────────▼──┐  │
   │          x_t (input)                       │  │
   └───────────────────────────────────────────┘  │
                           │                       │
                        ┌──▼──┐                   │
                        │ o_t │                   │
                        └──▼──┘                   │
                           ×─────────────────────▼─→ h_t
                        tanh(C_t)
```

### **LSTM Gates**

### **Vanilla RNN**

- **Hidden State**: `h_t = tanh(W_hh·h_{t-1} + W_ih·x_t + b)`
- **Output**: `y_t = W_hy·h_t + b_y`
- **Problem**: Vanishing gradients for long sequences

### **LSTM (Long Short-Term Memory)**

- **Forget Gate**: `f_t = σ(W_f·[h_{t-1}, x_t] + b_f)`
- **Input Gate**: `i_t = σ(W_i·[h_{t-1}, x_t] + b_i)`
- **Candidate**: `C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C)`
- **Cell State**: `C_t = f_t * C_{t-1} + i_t * C̃_t`
- **Output Gate**: `o_t = σ(W_o·[h_{t-1}, x_t] + b_o)`
- **Hidden State**: `h_t = o_t * tanh(C_t)`

### **GRU (Gated Recurrent Unit)**

- **Reset Gate**: `r_t = σ(W_r·[h_{t-1}, x_t])`
- **Update Gate**: `z_t = σ(W_z·[h_{t-1}, x_t])`
- **New Memory**: `h̃_t = tanh(W·[r_t * h_{t-1}, x_t])`
- **Hidden State**: `h_t = (1-z_t) * h_{t-1} + z_t * h̃_t`

### **When to Use**

- **Vanilla RNN**: Short sequences (<10 steps)
- **LSTM**: Long sequences, need long-term dependencies
- **GRU**: Efficient alternative to LSTM, 25% fewer parameters
- **Bidirectional**: When full context available (not real-time)

## ⚖️ **4. BIAS-VARIANCE TRADEOFF**

### **Mathematical Definition**

`Total Error = Bias² + Variance + Irreducible Error`

### **Symptoms & Solutions**

| Problem           | Train Acc | Val Acc | Gap   | Solution           |
| ----------------- | --------- | ------- | ----- | ------------------ |
| **High Bias**     | Low       | Low     | Small | ↑ Model complexity |
| **High Variance** | High      | Low     | Large | ↑ Regularization   |
| **Good Fit**      | Medium    | Medium  | Small | ✓ Keep current     |

### **Bias vs Variance**

- **High Bias**: Underfitting, model too simple
- **High Variance**: Overfitting, model too complex
- **Goal**: Minimize total error, not individual components

## 🔄 **5. TRANSFER LEARNING**

### **Strategies**

1. **Feature Extraction**: Freeze pretrained weights, train classifier only
2. **Fine-tuning**: Unfreeze all/some layers, use lower LR for early layers
3. **Progressive Unfreezing**: Gradually unfreeze layers during training

### **When to Use**

- **Small dataset + similar domain** → Feature extraction
- **Large dataset + different domain** → Fine-tuning
- **Very small dataset** → Feature extraction + data augmentation

### **Learning Rates**

- **Pretrained layers**: 1e-5 to 1e-4
- **New classifier**: 1e-3 to 1e-2
- **Discriminative LR**: Early layers < Later layers

## 🛠️ **6. OPTIMIZATION & TRAINING**

### **Optimizers - Complete Equations**

#### **1. Stochastic Gradient Descent (SGD)**

```
Basic SGD:
w_{t+1} = w_t - α∇L(w_t)

SGD with Momentum:
v_t = βv_{t-1} + ∇L(w_t)
w_{t+1} = w_t - αv_t

where:
- α = learning rate
- β = momentum coefficient (typically 0.9)
- v_t = velocity (momentum term)
```

#### **2. Adagrad (Adaptive Gradient)**

```
G_t = G_{t-1} + ∇L(w_t)²   (element-wise)
w_{t+1} = w_t - (α/√(G_t + ε)) · ∇L(w_t)

where:
- G_t = accumulated squared gradients
- ε = small constant (1e-8) for numerical stability
- Problem: G_t grows monotonically → learning rate → 0
```

#### **3. RMSprop (Root Mean Square Propagation)**

```
G_t = γG_{t-1} + (1-γ)∇L(w_t)²   (element-wise)
w_{t+1} = w_t - (α/√(G_t + ε)) · ∇L(w_t)

where:
- γ = decay rate (typically 0.9)
- Fixes Adagrad's problem with exponential moving average
```

#### **4. Adam (Adaptive Moment Estimation)**

```
m_t = β₁m_{t-1} + (1-β₁)∇L(w_t)        [1st moment - mean]
v_t = β₂v_{t-1} + (1-β₂)∇L(w_t)²       [2nd moment - variance]

Bias correction:
m̂_t = m_t/(1-β₁ᵗ)
v̂_t = v_t/(1-β₂ᵗ)

Update:
w_{t+1} = w_t - α(m̂_t/√(v̂_t + ε))

Default hyperparameters:
- α = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 1e-8
```

#### **5. AdamW (Adam with Weight Decay)**

```
m_t = β₁m_{t-1} + (1-β₁)∇L(w_t)
v_t = β₂v_{t-1} + (1-β₂)∇L(w_t)²

m̂_t = m_t/(1-β₁ᵗ)
v̂_t = v_t/(1-β₂ᵗ)

w_{t+1} = w_t - α(m̂_t/√(v̂_t + ε)) - αλw_t

where:
- λ = weight decay coefficient (decoupled from gradient)
- More effective than L2 regularization in Adam
```

#### **6. Adadelta**

```
E[g²]_t = ρE[g²]_{t-1} + (1-ρ)∇L(w_t)²
RMS[g]_t = √(E[g²]_t + ε)

Δw_t = -(RMS[Δw]_{t-1}/RMS[g]_t) · ∇L(w_t)
w_{t+1} = w_t + Δw_t

E[Δw²]_t = ρE[Δw²]_{t-1} + (1-ρ)Δw_t²
RMS[Δw]_t = √(E[Δw²]_t + ε)

where:
- ρ = decay constant (typically 0.95)
- No learning rate needed!
```

#### **7. Nadam (Nesterov Adam)**

```
m_t = β₁m_{t-1} + (1-β₁)∇L(w_t)
v_t = β₂v_{t-1} + (1-β₂)∇L(w_t)²

m̂_t = m_t/(1-β₁ᵗ)
v̂_t = v_t/(1-β₂ᵗ)

Nesterov momentum:
w_{t+1} = w_t - α[β₁m̂_t + ((1-β₁)/(1-β₁ᵗ))∇L(w_t)]/√(v̂_t + ε)
```

#### **8. RAdam (Rectified Adam)**

```
Standard Adam updates with rectification term:

ρ∞ = 2/(1-β₂) - 1
ρ_t = ρ∞ - 2tβ₂ᵗ/(1-β₂ᵗ)

if ρ_t > 4:
    r_t = √[(ρ_t-4)(ρ_t-2)ρ∞]/[(ρ∞-4)(ρ∞-2)ρ_t]
    w_{t+1} = w_t - αr_t(m̂_t/√(v̂_t + ε))
else:
    w_{t+1} = w_t - αm̂_t

where r_t is the rectification term that stabilizes early training
```

### **Optimizer Comparison Table**

| Optimizer        | Learning Rate | Momentum      | Adaptive LR    | Memory | Best For                                 |
| ---------------- | ------------- | ------------- | -------------- | ------ | ---------------------------------------- |
| **SGD**          | Manual        | Optional      | ❌             | Low    | Simple problems, fine-tuning             |
| **SGD+Momentum** | Manual        | ✅            | ❌             | Low    | CNNs, when convergence matters           |
| **Adagrad**      | Adaptive      | ❌            | ✅             | Medium | Sparse data, early stopping              |
| **RMSprop**      | Adaptive      | ❌            | ✅             | Medium | RNNs, non-stationary objectives          |
| **Adam**         | Adaptive      | ✅            | ✅             | High   | **Default choice**, most problems        |
| **AdamW**        | Adaptive      | ✅            | ✅             | High   | Transformers, when regularization needed |
| **Nadam**        | Adaptive      | ✅ (Nesterov) | ✅             | High   | When faster convergence needed           |
| **RAdam**        | Adaptive      | ✅            | ✅ (Rectified) | High   | Unstable early training                  |

### **Quick Optimizer Selection Guide**

```
Problem Type?
├── Small dataset/Simple model → SGD + Momentum
├── Computer Vision (CNN) → SGD + Momentum OR Adam
├── NLP/Transformers → AdamW
├── RNN/LSTM → RMSprop OR Adam
├── Unstable training → RAdam
├── Need fast convergence → Nadam
└── Default/Unsure → Adam
```

### **Learning Rate Guidelines**

- **SGD**: 0.01 - 0.1
- **Adam/AdamW**: 0.001 - 0.003
- **RMSprop**: 0.001
- **Rule of thumb**: Start with defaults, then tune
- **Scheduling**: Reduce LR when loss plateaus

### **Regularization**

- **L1 (Lasso)**: `Loss + λ|w|` - Sparsity, feature selection
- **L2 (Ridge)**: `Loss + λw²` - Weight decay, smooth
- **Dropout**: Randomly zero neurons during training
- **Batch Norm**: Normalize layer inputs, stabilizes training

### **Training Techniques**

- **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_(params, 1.0)`
- **Early Stopping**: Stop when validation loss stops improving
- **Learning Rate Scheduling**: ReduceLROnPlateau, Cosine annealing

### **Hyperparameters**

- **Batch Size**: 32, 64, 128
- **Hidden Units**: 64, 128, 256
- **Dropout**: 0.2-0.5
- **Weight Decay**: 1e-4 to 1e-5

## 📊 **7. PRACTICAL TIPS**

### **Model Selection Guide**

```
Data Type?
├── Tabular → ANN
├── Images → CNN
└── Sequential
    ├── Short → RNN
    ├── Long + Accuracy → LSTM
    └── Long + Speed → GRU
```

### **PyTorch Training Loop**

```python
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
```

### **Common Issues & Fixes**

- **Exploding Gradients** → Gradient clipping
- **Vanishing Gradients** → LSTM/GRU, skip connections
- **Overfitting** → Dropout, regularization, more data
- **Underfitting** → Larger model, train longer
- **Slow Convergence** → Higher LR, better optimizer

### **Layer Counting Rule**

Count only layers with **learnable parameters**:

- ✅ nn.Linear, nn.Conv2d, nn.LSTM
- ❌ ReLU, Dropout, MaxPool (no parameters)

## 🤖 **8. MODERN ARCHITECTURES**

### **Autoencoders**

```
Encoder:    Input → Hidden (Bottleneck) → Latent Space
Decoder:    Latent Space → Hidden → Reconstructed Output
Goal: Minimize reconstruction loss: L = ||x - x̂||²
```

- **Use Cases**: Dimensionality reduction, denoising, anomaly detection
- **Variational AE (VAE)**: Adds probabilistic latent space

### **Attention Mechanisms**

- **Formula**: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- **Self-Attention**: Q, K, V all from same input
- **Purpose**: Focus on relevant parts of input sequence

### **Transformer Architecture**

```
Input → Positional Encoding → Multi-Head Attention → LayerNorm → FFN → Output
                                    ↑                               ↑
                              (Residual Connection)        (Residual Connection)
```

- **Key Innovation**: Attention is all you need (no RNNs!)
- **Benefits**: Parallelizable, handles long sequences better

### **Generative Models**

- **GAN**: Generator vs Discriminator (adversarial training)
- **VAE**: Variational Autoencoders (probabilistic approach)
- **Diffusion**: Gradual denoising process

## 🧪 **9. MODERN TECHNIQUES**

### **Layer Normalization**

- **Formula**: `LN(x) = γ((x-μ)/σ) + β` (normalize across features)
- **vs BatchNorm**: Works per sample, better for RNNs/Transformers

### **Residual Connections**

- **Formula**: `y = F(x) + x` (skip connection)
- **Purpose**: Solves vanishing gradients in very deep networks

### **Advanced Regularization**

- **Label Smoothing**: Soft targets instead of hard 0/1
- **Mixup**: Train on weighted combinations of samples
- **Cutout**: Random masking of image patches

### **Exam Memory Aids**

- **BIAS = BAD + SIMPLE** vs **VARIANCE = VOLATILE + COMPLEX**
- **CNN**: Spatial patterns, weight sharing, hierarchical
- **RNN**: Sequential data, hidden state memory
- **LSTM**: Gates control information flow
- **Transfer Learning**: Start with pretrained, adapt to new task
- **Attention**: Focus mechanism, "what to pay attention to"
- **Transformers**: All attention, highly parallelizable

## 🎮 **10. REINFORCEMENT LEARNING**

### **RL Agent Types**

```
1. Model-Free Agents:    Learn directly from experience
   • Value-based:        Q-Learning, DQN
   • Policy-based:       REINFORCE, Actor-Critic
   • Actor-Critic:       Combines value + policy

2. Model-Based Agents:   Learn environment model first
   • Planning-based:     Use model to plan actions
   • Dyna-Q:            Mix model learning + direct experience
```

### **Basic RL Mechanism**

```
Environment ←→ Agent
     ↓           ↑
   State(s)   Action(a)
     ↓           ↑
  Reward(r) → Learning

Goal: Maximize cumulative reward Σγᵗrₜ
```

- **MDP Components**: States (S), Actions (A), Rewards (R), Transitions (P), Discount (γ)
- **Policy π(a|s)**: Probability of action a in state s
- **Value Function V(s)**: Expected return from state s
- **Q-Function Q(s,a)**: Expected return from state s, action a

### **Q-Learning Algorithm**

```
Q-Table Update:
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
                      ↑              ↑        ↑
                   Learning    Best future   Current
                     Rate       Q-value      Q-value
```

**Steps:**

1. Observe state s
2. Choose action a (ε-greedy: explore vs exploit)
3. Execute action, get reward r and next state s'
4. Update Q(s,a) using Bellman equation
5. Repeat until convergence

### **Deep Q-Learning (DQN)**

```
Neural Network Architecture:
State → [Conv/FC Layers] → Q-values for all actions
  s                          Q(s,a₁), Q(s,a₂), ..., Q(s,aₙ)
```

**Key Innovations:**

- **Experience Replay**: Store (s,a,r,s') in buffer, sample random batches
- **Target Network**: Separate network for stable targets (update periodically)
- **Loss Function**: `L = (r + γ max Q_target(s',a') - Q(s,a))²`

### **DQN Algorithm**

```python
# Pseudocode
for episode in episodes:
    state = env.reset()
    for step in steps:
        # ε-greedy action selection
        if random() < ε:
            action = random_action()
        else:
            action = argmax(Q(state))

        next_state, reward, done = env.step(action)
        buffer.store(state, action, reward, next_state, done)

        # Train network
        if len(buffer) > batch_size:
            batch = buffer.sample(batch_size)
            target = r + γ * max(Q_target(s'))
            loss = (target - Q(s,a))²
            optimize(loss)

        # Update target network periodically
        if step % C == 0:
            Q_target = Q.copy()
```

### **Deep RL Techniques**

**1. Policy Gradient Methods**

- **REINFORCE**: `∇J(θ) = Σ ∇log π(a|s) × R`
- **Actor-Critic**: Actor (policy) + Critic (value function)
- **A3C**: Asynchronous Actor-Critic with multiple workers

**2. Advanced Value Methods**

- **Double DQN**: Separate action selection and evaluation
- **Dueling DQN**: Split Q(s,a) = V(s) + A(s,a)
- **Prioritized Experience Replay**: Sample important transitions more

**3. Policy Optimization**

- **PPO**: Proximal Policy Optimization (stable policy updates)
- **TRPO**: Trust Region Policy Optimization
- **SAC**: Soft Actor-Critic (maximum entropy RL)

### **RL Problem Types**

- **Discrete Actions**: DQN, Q-Learning
- **Continuous Actions**: Policy gradients, Actor-Critic
- **Multi-Agent**: Independent learners, centralized training
- **Partial Observability**: POMDP, recurrent policies

### **Exploration Strategies**

- **ε-greedy**: Random action with probability ε
- **UCB**: Upper Confidence Bound
- **Thompson Sampling**: Bayesian exploration
- **Curiosity-driven**: Intrinsic motivation

### **RL Visual Architecture**

```
DQN Architecture:
Input State → CNN/FC → Hidden Layers → Q-values Output
   84×84×4      Conv      Dense         |A| neurons
    ↓            ↓         ↓              ↓
  Atari        Feature   Value       Action Selection
  Frames      Extraction Function    (argmax or ε-greedy)

Actor-Critic:
State → Shared Network → Actor (Policy π)
                      → Critic (Value V)
```

---

**🎓 Remember**: Understand the WHY behind each technique, not just the equations!
