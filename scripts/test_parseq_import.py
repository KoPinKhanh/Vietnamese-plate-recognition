import torch
import sys

print("Testing torch.hub for baudm/parseq...")
try:
    # Test torch hub
    parseq = torch.hub.load('baudm/parseq', 'parseq_tiny', pretrained=True, trust_repo=True)
    print("PARSeq Tiny loaded successfully via torch.hub!")
    print(parseq)
except Exception as e:
    print(f"torch.hub failed: {e}")
    sys.exit(1)
