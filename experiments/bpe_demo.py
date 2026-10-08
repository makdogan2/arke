from collections import Counter

metin = open("data/mix.txt", encoding="utf-8").read()[:200_000]

# başlangıçta her karakter ayrı bir token
tokenlar = list(metin)

for adim in range(20):
    ciftler = Counter(zip(tokenlar, tokenlar[1:]))
    if not ciftler:
        break
    (a, b), sayi = ciftler.most_common(1)[0]
    print(f"{adim+1:2d}. birleşme: {a!r} + {b!r} -> {a+b!r}  ({sayi} kez)")

    # bu çifti tek token olarak birleştir
    yeni, i = [], 0
    while i < len(tokenlar):
        if i < len(tokenlar)-1 and tokenlar[i] == a and tokenlar[i+1] == b:
            yeni.append(a + b)
            i += 2
        else:
            yeni.append(tokenlar[i])
            i += 1
    tokenlar = yeni

print("\nkarakter sayısı:", len(metin))
print("token sayısı   :", len(tokenlar))
print("sıkışma oranı  :", round(len(metin)/len(tokenlar), 2))