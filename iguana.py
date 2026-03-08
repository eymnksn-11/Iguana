import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
import urllib.request
from collections import Counter

# ==========================================
# 1. OPTIMIZER: IGuAna
# ==========================================
class IGuAna(Optimizer):
    def __init__(self, params, base_lr=0.00001, beta=0.9, k_hedge=0.1, boost_scale=0.01, eps=1e-8):
        """
        IGuAna: Inverse Gradient Unbound Accelerated Network Adaptor.
        
        Args:
            base_lr: Base learning rate (0.00001 recommended for high boost scales).
            beta: Exponential moving average coefficient for momentum.
            k_hedge: Safety coefficient to prevent instability during high acceleration.
            boost_scale: Scaling factor for the inverse variance boost.
            eps: Term added to the denominator to improve numerical stability.
        """
        if base_lr < 0.0:
            raise ValueError(f"Invalid learning rate: {base_lr}")
        
        defaults = dict(base_lr=base_lr, beta=beta, k_hedge=k_hedge, 
                        boost_scale=boost_scale, eps=eps)
        super(IGuAna, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_boost = 0.0
        total_a_coeff = 0.0
        param_count = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                beta = group['beta']
                state['step'] += 1

                # Update momentum (EMA)
                exp_avg.mul_(beta).add_(grad, alpha=1 - beta)

                # Variance-based Boost Calculation
                # Higher stability (lower variance) leads to higher acceleration.
                v_var = torch.var(exp_avg) if exp_avg.numel() > 1 else torch.tensor(0.0, device=p.device)
                boost = 1.0 / (v_var + group['eps'])
                
                # Global safety clamp
                boost = torch.clamp(boost, max=1e8)

                # Hedge mechanism: Braking based on gradient norm to prevent instability
                grad_norm = torch.norm(grad)
                a_coeff = 1.0 / (1.0 + group['k_hedge'] * grad_norm)

                # Dynamic update scaling
                update_scale = group['base_lr'] * (boost * group['boost_scale']) * a_coeff
                
                # Parameter update
                p.sub_(exp_avg, alpha=update_scale.item())

                # Statistics collection
                total_boost += boost.item()
                total_a_coeff += a_coeff.item()
                param_count += 1

        avg_boost = total_boost / (param_count + 1e-8)
        avg_acoeff = total_a_coeff / (param_count + 1e-8)
        
        return loss, avg_boost, avg_acoeff

# ==========================================
# 2. DATA PREPARATION (Wikitext-2)
# ==========================================
print("Downloading and preparing Wikitext-2 dataset...")
url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
text = urllib.request.urlopen(url).read().decode('utf-8').split()[:50000]

vocab = Counter(text)
word2idx = {word: i for i, (word, _) in enumerate(vocab.items())}
data = torch.tensor([word2idx[w] for w in text], dtype=torch.long)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    return data.view(bsz, -1).t().contiguous()

batch_size = 32
bptt = 35 
train_data = batchify(data, batch_size)
vocab_size = len(word2idx)

# ==========================================
# 3. MODEL ARCHITECTURE (Transformer)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MiniTransformer(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers):
        super().__init__()
        self.pos_encoder = PositionalEncoding(ninp)
        self.encoder = nn.Embedding(ntoken, ninp)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(ninp, ntoken)
        self.ninp = ninp

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return self.decoder(output)

# Hyperparameters
emsize = 128
nhid = 256
nlayers = 2
nhead = 2
model = MiniTransformer(vocab_size, emsize, nhead, nhid, nlayers)
criterion = nn.CrossEntropyLoss()

# Initialize IGuAna Optimizer
optimizer = IGuAna(model.parameters(), base_lr=0.00001, k_hedge=0.1)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

print(f"\nTraining Started (Vocab Size: {vocab_size})")
print("-" * 110)

epochs = 20
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.
    epoch_boost = 0.
    epoch_acoeff = 0.
    steps = 0
    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        src_mask = generate_square_subsequent_mask(data.size(0))
            
        optimizer.zero_grad()
        output = model(data, src_mask)
        
        loss = criterion(output.view(-1, vocab_size), targets)
        loss.backward()
        
        _, avg_boost, avg_acoeff = optimizer.step()
        
        total_loss += loss.item()
        epoch_boost += avg_boost
        epoch_acoeff += avg_acoeff
        steps += 1

    cur_loss = total_loss / steps
    ppl = math.exp(cur_loss) if cur_loss < 20 else float('inf') 
    
    print(f"Epoch {epoch:2d} | Loss: {cur_loss:.4f} | PPL: {ppl:8.2f} | Avg Boost: {epoch_boost/steps:12.2f} | Avg a-Coeff: {epoch_acoeff/steps:.4f}")

print("-" * 110)
print("Training Complete.")
