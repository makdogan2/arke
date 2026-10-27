# Arke Defteri

Aşamaların kaydı: şu an ne yapıyoruz, sırada ne var, ne bitti, neye karar verdik.
Her aşamanın sonunda güncellenir.

## Şu anki aşama

### 2. Yeni veriyle ilk model (`arke0_web`), 98M
- **Ne:** `l` boyutu (12 katman, 768 genişlik, 12 head; vocab 16384 ile 97.8M parametre).
- **Neden bu boyut:** Model boyutu deneyinde `l` (91.5M), `m`'yi az veriyle bile geçti (val 3.365'e karşı 3.545). 33.7M'lik ilk plan Chinchilla'yı fazla katı uyguluyordu.
- **Plan C:** ~2.0B token eğitim, veri 1,069M token olduğu için ~1.87 epoch. 2.0B / 97.8M ≈ 20 token/parametre, yani Chinchilla dengesi.
- **Ölçüldü (27 Ekim):** `l` compile'lı **95k tok/s**, GPU belleği en fazla ~10.8 GB. 2.0B token ≈ 5.9 saat, eval ile **~6.3 saat** saf eğitim.
- **Oturumlara bölünerek çalışacak:** `--until 08:00` ya da `--hours 3` ile kendisi kaydedip durur. Ctrl+C de kaydedip durur. `--resume` kaldığı adımdan devam eder.
- **Kod:** Yeni kod gerekmiyor. Ama önerim, eğitime başlamadan önce Aşama 3'ü bitirmek: o zaman TR ve EN loss'u eğitimin ilk adımından itibaren ayrı ayrı loglanır.
- **Komutlar:**
  - İlk oturum: `.venv\Scripts\python.exe train.py --name arke0_web --n_layer 12 --n_embd 768 --n_head 12 --max-iters 122000 --until 08:00`. 2.0B token / 16,384 ≈ 122,000 adım.
  - Sonraki oturumlar: `.venv\Scripts\python.exe train.py --name arke0_web --resume --hours 3`.
- **Kontrol:** `runs/arke0_web.json`'da val loss sonda yükselmiyor olmalı (ezberleme yok). `sample.py --name arke0_web` ile TR ve EN metin.
- **Karşılaştırma:** Token başına loss farklı tokenizer'lar arasında kıyaslanamaz. Arke-0 ile karşılaştırmak için loss'u karakter/token oranına bölüp "karakter başına loss"a çevireceğiz (Arke-0: 1.100).

## Sıradaki aşamalar

### 3. Dil başına val loss (önerim: 98M eğitiminden önce)
- **Ne:** Val setini TR ve EN olarak ayrı ölçmek.
- **Neden:** Hedefimiz "İngilizce Türkçe kadar iyi". Şu an tek bir karışık val sayısı var; hangi dilin geride kaldığını göremiyoruz.
- **Kod:** `data_prep.py` val'i kaynak başına ayrı dosyalara da yazacak, `train.py` loss'u dil başına raporlayacak. Kodu sohbette vereceğim, sen yazacaksın.
- **Veri tekrar indirilmez:** `data_prep.py` tekrar çalışınca bütün aşamaları atlar, sadece son aşama yeni kodla `.bin` dosyalarını yazar (~1 dakika).

### 4. Kendi modelimizi quantize etmek (8 / 4 / 2 bit)
- **Ne:** `arke0_web`'in linear katman ağırlıklarını 8, 4 ve 2 bite sıkıştırıp geri açmak. Ölçek grup başına (ör. 64 ağırlıkta bir), quantization kodunu kütüphanesiz kendimiz yazacağız.
- **Neden:** Arkadaşının kullandığı araçların (GGUF, Ollama) içinde ne olduğunu anlamak. Parametre sayısı değişmez; sadece her parametre daha az bitle saklanır.
- **Ölçülecek:** Her bit sayısında dosya boyutu (98M: bf16 196 MB, int8 ~98 MB, int4 ~49 MB, int2 ~24 MB, ölçekler hariç) ve val loss'taki artış. Bir de üretilen metin örnekleri.
- **Önkoşul:** Aşama 2'deki eğitilmiş model.

### Daha sonra (sıra değişebilir)
- `generate` `<|endoftext|>` görünce dursun (şu an belge bitince yeni bir belgeye geçip devam ediyor).
- Türkçeye uygun pre-tokenizer (ekleri daha iyi bölmek için). Ayrı bir tokenizer deneyi.
- Yeni veriyle model boyutu deneyi (xs/s/m/l) ve README'ye ölçekleme bölümü (`plot_scaling.py` hazır).
- Arke-1: sohbet edebilen asistan (roadmap'teki son madde).

## Biten aşamalar

| Tarih | Aşama | Commit |
|---|---|---|
| 24 Ekim | Resume iskeleti, `data_prep.py`, `textclean.py` (WIP) | `11bca9a` |
| 25 Ekim | `main` ile birleşme; checkpoint + resume; tekrar filtresi; 15 test | `f413556` |
| 25 Ekim | `train.bin` + memmap, fused AdamW, `torch.compile` (207k → 251k tok/s) | `db721ad` |
| 25 Ekim | UTF-8 çıktı; resume'da eval/save aralığı (kesintisiz koşuyla birebir aynı) | `a10b0cf` |
| 27 Ekim | README yeni pipeline'a göre güncellendi | `f12b75e` |
| 27 Ekim | Oturumlu eğitim: `--hours`, `--until`, güvenli Ctrl+C; her duruşta `runs/` logu güncelleniyor. İlk kez yeni çalışma şekliyle yazıldı (kodu Mehmet yazdı). Test: süreyle durma 723 → resume; Ctrl+C ile 315 → 857 → 1000, üç oturum logda doğru birleşti. | `94a39bb` |
| 27 Ekim | Gerçek veri: wiki 191 MB + TR 2,102 MB + EN 2,142 MB → **train 1,069M token**, val 10.7M. Kaynaklar (token): wiki %5, tr %46, en %50. Tokenizer 16384: TR 3.88, EN 3.98 karakter/token (eski 8192: 3.32 / 3.34). Dedup: TR 22, EN 482 kopya. Süre ~23 dk (bir `PermissionError` kesintisi dahil). | — (veri git'te değil) |

Daha önce (`main`'de, 17 Ekim): model boyutu argümanları ve `runs/` logu, referans bölümü temizliği, `plot_scaling.py`.

## Kararlar

| Tarih | Karar | Neden |
|---|---|---|
| 24 Ekim | Veri: FineWeb2-HQ (TR) + FineWeb-Edu (EN) + mevcut Wikipedia | Açık lisans (ODC-By), kaliteye göre filtrelenmiş. CulturaX erişim izni istiyor ve daha eski. |
| 24 Ekim | TR/EN %50/%50, +2 GB yeni metin | İngilizce Türkçe kadar iyi olsun. |
| 24 Ekim | Tokenizer yeniden eğitildi, vocab 16384 | İki dil dengeli sıkışsın: smoke testte TR 3.83, EN 3.93 karakter/token (eski 8192'de ikisi de ~3.33). |
| 25 Ekim | `triton-windows` 3.7 kuruldu | `torch.compile` için; torch 2.11 ↔ Triton 3.7. |
| 27 Ekim | Model boyutu deneyi README'ye şimdilik eklenmedi | Eski veriyle yapıldı; yeni veriyle tekrarlanınca anlamlı bir bölüm olur. `runs/` commit dışında kalıyor. |
| 27 Ekim | Veri hedefi ikiye katlandı: TR 2.1 GB, EN 2.14 GB (toplam ~4.4 GB, ~1.1B token) | 98M'lik model için Chinchilla dengesi (~2.0B token eğitim, ~1.8 epoch). Türkçe HQ veri bol (105 dosya). |
| 27 Ekim | Dosyalar `replace_retry` ile yerine konuyor (`data_prep.py`, `train.py`) | İlk gerçek koşu 12. dakikada `PermissionError: Access is denied` ile durdu: `state.json.tmp` → `state.json` adı değiştirilirken Windows'ta başka bir program (büyük ihtimalle Defender ya da arama indeksleyicisi) dosyayı bir anlığına açık tutuyordu. Artık kısa bekleyip 10 kez deniyor. Veri kaybı olmadı: eski `state.json` sağlamdı, sadece son shard yeniden işlenecek. |
| 27 Ekim | Bir sonraki model 98M (`l` boyutu), 33.7M değil | Model boyutu deneyinde büyük model az veriyle bile kazandı. 1B+ sıfırdan eğitim bu kartta pratik değil (1B ≈ 25 gün, 16 GB'a sığmıyor); büyük modeller için yol ince ayar (QLoRA). |
| 27 Ekim | Yeni çalışma şekli: kodu ben sohbette eksiksiz veririm, sen VS Code'da yazarsın, "commit" deyince commit | Kodu yazarak öğrenmek ve kontrolü elde tutmak. |
