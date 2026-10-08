from collections import Counter
import regex as re

metin = open("data/mix.txt", encoding="utf-8").read()[:200_000]

# GPT-2'nin ön-parçalama deseni: boşluk kelimenin başına gider
DESEN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+""")

parcalar = DESEN.findall(metin)
print("ilk 15 parça:", parcalar[:15])

# her benzersiz parça, kaç kez geçtiğiyle birlikte
frekans = Counter(parcalar)
kelimeler = {k: list(k) for k in frekans}


for adim in range(30):
    ciftler = Counter()
    for kelime, sayi in frekans.items():
        parcalanmis = kelimeler[kelime]
        for cift in zip(parcalanmis, parcalanmis[1:]):
            ciftler[cift] += sayi          # sıklıkla ağırlıklandır
    if not ciftler:
        break
    (a, b), sayi = ciftler.most_common(1)[0]
    print(f"{adim+1:2d}. {a!r} + {b!r} -> {a+b!r}  ({sayi})")

    for kelime in kelimeler:
        eski, yeni, i = kelimeler[kelime], [], 0
        while i < len(eski):
            if i < len(eski)-1 and eski[i] == a and eski[i+1] == b:
                yeni.append(a + b); i += 2
            else:
                yeni.append(eski[i]); i += 1
        kelimeler[kelime] = yeni

toplam_token = sum(len(kelimeler[k]) * n for k, n in frekans.items())
print("\nsıkışma oranı:", round(len(metin) / toplam_token, 2))