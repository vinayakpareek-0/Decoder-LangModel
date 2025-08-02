import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# --- --- --- --- --- ---

torch.manual_seed(1337)

# That sakesphere text file:
with open('/content/input.txt','r', encoding='utf-8') as f: text=f.read()
chars= sorted(list(set(text)))
vocab_size=len(chars)

# Converts str to int ans vice-versa:
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# Splitting train and test data:

data= torch.tensor(encode(text) , dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Loading Data:
def get_batch(split):
    data= train_data if split=='train' else val_data
    ix = torch.randint(len(data)-block_size , (batch_size,))
    x= torch.stack([ data[i:i+block_size] for i in ix])
    y= torch.stack([ data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
# print(get_batch('val'))

print("Chai peeni hai merko 😔")

""" ---   ---   ---    ---   ---   --- """

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ---   ---   ---    ---   ---   ---

""" Single head of Self-attention: 🙂 Like both of us """

class Head(nn.Module):
    def __init__(self , head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size , bias = False)
        self.query= nn.Linear(n_embd, head_size , bias = False)
        self.value = nn.Linear(n_embd, head_size , bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout= nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)

        """ compute attention scores ("affinities") """
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        """ perform the weighted aggregation of the values """
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel using different linear projections using""" #  nn.ModuleList
    def __init__(self , num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self , n_embd , n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        self.blocks=nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
        self.ln= nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        # self.apply(self._init_weights    )

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self , idx , targets=None):
        B,T= idx.shape
        tk_emb= self.token_embedding_table(idx) # (B,T,C)
        pos_emb= self.position_embedding_table(torch.arange(T,device=device)) # (T,C)
        x = tk_emb + pos_emb # (B,T,C)
        self.blocks(x) # (B,T,C)
        self.ln(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits , loss

    def generate(self, idx, max_new_tokens):
       def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop idx to the last block_size tokens
            logits, loss = self(idx_cond) # get the predictions
            logits = logits[:, -1, :] # becomes (B, C)""" # focus only on the last time step"""
            probs = F.softmax(logits, dim=-1) # (B, C) """# apply softmax to get probabilities"
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) -> """# sample from the distribution"
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) -> """# append sampled index to the running sequence"
        return idx

model = LanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

""" Generate from trained model  """
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))