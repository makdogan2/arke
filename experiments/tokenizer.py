from collections import Counter
from pathlib import Path
import torch

KOK = Path(__file__).parent.parent        # repo root
MIX = KOK / "data" / "mix.txt"
CIKTI = KOK / "data" / "char.pt"

ESIK = 200 # bu sayıdan az geçen karakterler elenecek

ZORUNLU = set(
    "abcçdefgğhıijklmnoöprsştuüvyz"
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
    "qwxQWX"
    "0123456789"
    " \n.,;:!?'\"()-–—/%&*+=[]{}<>@#$_"
)  

metin = open(MIX, encoding="utf-8").read()
sayac = Counter(metin)
toplam = sum(sayac.values())

print("toplam karakter:", len(metin))
print("benzersiz karakter:", len(sayac))

# --- 1) kapsama analizi ---
birikim = 0
for i, (k, n) in enumerate(sayac.most_common(), 1):
    birikim += n
    if i in (50, 100, 150, 200):
        print(f"ilk {i} karakter -> %{100*birikim/toplam:.4f}")

# --- 2) neyi atıyoruz ---
atilan = [(k, n) for k, n in sayac.most_common() if n < ESIK]
print("atılan karakter sayısı:", len(atilan))
print("atılanların en sıkı:", atilan[:25])

# --- 3) sözlüğü kur ---
tutulan = {k for k, n in sayac.items() if n >= ESIK} | (ZORUNLU & set(metin))
kars = ["<unk>"] + sorted(tutulan)
stoi = {k: i for i, k in enumerate(kars)}
itos = {i: k for k, i in stoi.items()}
vocab_size = len(kars)
print("vocab_size:", vocab_size)

def encode(s): return [stoi.get(c, 0) for c in s]
def decode(l): return "".join(itos[i] for i in l)

print("kayıpsızlık testi:", decode(encode("Merhaba Arke, how are you?")))

# --- 4) kodla ve kaydet ---
kodlu = encode(metin)
print(f"unk oranı: %{100 * kodlu.count(0) / len(kodlu):.4f}")

veri = torch.tensor(kodlu, dtype=torch.uint16)
n = int(0.9 * len(veri))
torch.save({"train": veri[:n], "val": veri[n:], "stoi": stoi, "itos": itos}, CIKTI)
print("kaydedildi:", CIKTI, "| token sayısı:", len(veri))