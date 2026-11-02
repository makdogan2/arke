# Packing 4-bit values: two per byte. First on 8 numbers, then a round trip on a million.
import torch

def pack4(q):
    """q: uint8 values in 0..15, even count. Element 2i goes to the low 4 bits of byte i, 2i+1 to the high 4 bits."""
    q = q.reshape(-1)
    assert q.numel() % 2 == 0 and int(q.max()) <= 15
    return q[0::2] | (q[1::2] << 4)

def unpack4(packed):
    low = packed & 0x0F                                     # 0x0F = 0000 1111: keep the low 4 bits
    high = packed >> 4                                      # shift the high 4 bits down
    return torch.stack([low, high], dim=-1).reshape(-1)     # interleave back: low0, high0, low1, high1, ...

# --- part 1: eight numbers by hand ---
q = torch.tensor([3, 12, 0, 15, 7, 9, 1, 8], dtype=torch.uint8)
packed = pack4(q)
print(f"q        {q.tolist()} | {q.numel()} bytes")
print(f"packed   {packed.tolist()} | {packed.numel()} bytes")
for i, b in enumerate(packed.tolist()):
    print(f"  byte {i}: {b:3} = {b:08b} | high {b >> 4:2} ({b >> 4:04b}) low {b & 15:2} ({b & 15:04b})")
print(f"unpacked {unpack4(packed).tolist()}")

# --- part 2: a million random 4-bit values, exact round trip ---
torch.manual_seed(0)
q = torch.randint(0, 16, (1_000_000,), dtype=torch.uint8)
packed = pack4(q)
print(f"\n{q.numel():,} values: {q.numel():,} bytes -> {packed.numel():,} bytes packed | "
      f"round trip equal: {torch.equal(unpack4(packed), q)}")