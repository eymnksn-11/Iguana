# IGuAna: Inverse Gradient Unbound Accelerated Network Adaptor

IGuAna is a high-velocity, adaptive first-order optimizer designed for deep learning architectures. While industry standards like Adam focus on dampening gradient noise through second-moment normalization, IGuAna introduces a radical shift: it leverages the inverse of gradient variance to reward directional stability with exponential acceleration.

# The Clarity Reward Hypothesis

Traditional optimizers treat variance as a signal to slow down. IGuAna operates on the "Clarity Reward" principle:High Variance: Uncertainty in the gradient direction leads to cautious, smaller steps.Low Variance: When the gradient path stabilizes (finding a "valley floor"), IGuAna triggers Unbound Acceleration, amplifying the update step by factors of up to $10^7$ to traverse plateaus instantly.

# Benchmark: IGuAna vs. Adam
Adam:

<img width="718" height="369" alt="adam-wikitext" src="https://github.com/user-attachments/assets/9c47a65b-8260-4e79-bb6b-459dd89c6167" />

IGuAna:

<img width="795" height="372" alt="13ppl-20e" src="https://github.com/user-attachments/assets/b9600c9b-71a0-4a83-a25f-c0073d6c4b47" />

The results show that IGuAna achieves near-convergence 30% faster than Adam and reaches a state of extreme precision (PPL 13) that standard methods fail to hit in comparable timeframes.

# How It Works

IGuAna governs the update rule through three distinct components:

The Thruster (Boost): Calculates $\frac{1}{Var(\text{exp\avg})}$. As gradients become consistent, the step size k-scales automatically.

The Hedge (Insurance): A safety coefficient, $a = \frac{1}{1 + k \cdot \|\nabla\|}$, that prevents "cliff-diving" during high-acceleration phases by monitoring the gradient norm.

The Chronos Memory: An exponential moving average (EMA) that ensures the "momentum of direction" is preserved, preventing oscillations.

# Installation & Quick Start
Simply drop the iguana.py into your project and replace your optimizer:

```python
from iguana import IGuAna

# Recommended: The "0.0001 Key" for Transformer stability
optimizer = IGuAna(
    model.parameters(), 
    base_lr=0.0001, 
    beta=0.9, 
    boost_scale=0.01, 
    k_hedge=0.1
)

# Training loop
loss.backward()
optimizer.step()
```

# Mathematical Intuition

Unlike Adam's update:

   $\Delta w = -\eta \frac{m_t}{\sqrt{v_t} + \epsilon}$

IGuAna utilizes:

   $\Delta w = -\eta \cdot \left( \frac{1}{\text{Var}(m_t) + \epsilon} \cdot \text{scale} \right) \cdot \text{Hedge}(\nabla) \cdot m_t$

Where $m_t$ is the momentum. This causes the optimizer to "rush" through stable valleys where $\text{Var}(m_t) \to 0$.



