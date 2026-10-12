import torch

parseq = torch.hub.load('baudm/parseq', 'parseq_tiny', pretrained=True, trust_repo=True).eval()

print("Attrs:", [a for a in dir(parseq) if not a.startswith('_')])
dummy = torch.randn(1, 3, 32, 128)
logits = parseq(dummy)
print("Logits shape:", logits.shape)
probs = logits.softmax(-1)
pred, conf = parseq.tokenizer.decode(probs)
print("Decoded:", pred, conf)
