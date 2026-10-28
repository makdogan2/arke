# Arke Defteri

Aşamaların kaydı: şu an ne yapıyoruz, sırada ne var, ne bitti, neye karar verdik.
Her aşamanın sonunda güncellenir.

## Şu anki aşama

> **Kaldığımız yer (28 Ekim):** Aşama 2 bitti: `arke0_web` eğitildi, ölçüldü, README'ye yazıldı (commit bekliyor). Sıradaki: Aşama 4, quantization. İlk karar: hangi checkpoint quantize edilecek (`arke0_web.pt` 118k mı, tam val'de biraz daha iyi olan `arke0_web_last.pt` 122k mı).

### 4. Kendi modelimizi quantize etmek (8 / 4 / 2 bit)
- **Ne:** `arke0_web`'in linear katman ağırlıklarını 8, 4 ve 2 bite sıkıştırıp geri açmak. Ölçek grup başına (ör. 64 ağırlıkta bir), quantization kodunu kütüphanesiz kendimiz yazacağız.
- **Neden:** Arkadaşının kullandığı araçların (GGUF, Ollama) içinde ne olduğunu anlamak. Parametre sayısı değişmez; sadece her parametre daha az bitle saklanır.
- **Ölçülecek:** Her bit sayısında dosya boyutu (98M: bf16 196 MB, int8 ~98 MB, int4 ~49 MB, int2 ~24 MB, ölçekler hariç) ve val loss'taki artış. Bir de üretilen metin örnekleri.
- **Ölçüm yöntemi:** val loss'u tam val setinde, sabit pencerelerle ölç (eğitimdeki 50 rastgele batch ±0.008 oynuyor; 8 bitin farkı bundan küçük olabilir). Referans (bf16, `arke0_web_last.pt`): val 3.1606, TR 0.8203/ch, EN 0.7916/ch.
- **Önkoşul:** Aşama 2'deki eğitilmiş model (hazır).

## Sıradaki aşamalar

### Daha sonra (sıra değişebilir)
- `textclean.py`: art arda boşlukları teke indir. Wikipedia'da temizlikten sonra 1M karakterde 614 çift boşluk ve 22 tane 5+ boşluk dizisi kalıyor (web: 0–5). Model bunu öğrenip bazen boşluk/satır sonu döngüsüne giriyor. Etkisi için veri ve eğitim yeniden gerekir.
- Eval gürültüsü: `estimate_loss` her seferinde 50 rastgele batch alıyor, ardışık eval'ler ±0.008 oynuyor; eğitimin sonundaki gerçek iyileşmeden (~0.002 / 1000 adım) büyük. "Best checkpoint" bu yüzden şansa kalıyor. Sabit pencereler ya da daha fazla batch.
- `sample.py`: sabit seed (aynı çıktıyı tekrar üretebilmek için), TR ile dengeli sayıda EN prompt, ekrana checkpoint'in kendi val'ini yaz (şu an `_last` için de `best_val` yazıyor).
- `generate` `<|endoftext|>` görünce dursun (şu an belge bitince yeni bir belgeye geçip devam ediyor).
- Sequence length warmup: eğitime 256 token bağlamla başlayıp sonra 512'ye geçmek. 512'nin başta yavaş öğrenmesini (aşağıdaki block_size kararı) atlatmanın bir yolu; ayrı bir deney.
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
| 27 Ekim | `replace_retry`: Windows'un bir anlığına kilitlediği dosyalarda yeniden deneme (`data_prep.py`, `train.py`). Test: hedef dosya açık tutulurken rename bekleyip başardı. | `62ba405` |
| 27 Ekim | Aşama 3, dil başına val loss: `data_prep.py` `val_tr.bin` (5.0M token, 3.88 karakter/token) ve `val_en.bin` (5.2M, 3.97) yazıyor, `meta.json`'a karakter sayılarını ekliyor. `train.py` her eval'de TR ve EN loss'u karakter başına da gösteriyor. Test (17M model, 300 adım, gerçek veri): adım 0'da 9.75 ≈ ln(16384); adım 300'de TR 1.777/ch, EN 1.605/ch. | `2d52a29` |
| 27 Ekim | `train.py --dropout`; 98M için dropout 0 ve block_size 256 seçildi (Kararlar'da). | `18aaa1e` |
| 28 Ekim | Aşama 2, `arke0_web` (97.8M, 12 katman, 768, 12 head, block 256, dropout 0): 122,000 adım, 2.0B token (1.87 epoch), 5 sa 47 dk, 103k tok/s, 27 Ekim 20:52'den itibaren iki oturumda. **Eğri:** val sona kadar düştü, 2. epoch'ta da; train–val farkı ~0.02'den ~0.03'e çıktı (hafif ezberleme, val'e zararı yok). **Tam val'de (sabit pencereler):** son checkpoint 122k val 3.1606, TR 0.8203/ch, EN 0.7916/ch; "best" 118k 3.1636, yani best seçimi gürültüden. **Arke-0 ile aynı metinde:** iki modelin de görmediği 32 Wikipedia makalesinde 1.110 → **0.854/ch (−%23)**; web TR 1.265 → 0.820, web EN 1.233 → 0.792 (−%35; bir kısmı alan farkı, Arke-0 web görmedi). **Örnekler:** TR ve EN dil bilgisi iyi, bilgi uyduruyor, tekrar ediyor, bazen boşluk döngüsüne giriyor (büyük ihtimalle wiki'deki boşluk dizilerinden). README'ye yazıldı. | — (commit bekliyor) |

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
| 27 Ekim | 98M için dropout 0 (Arke-0'dan kalan 0.2 yerine) | 98M, 1500 adım: val 5.340 → **4.993** (block 256), 5.500 → 5.216 (block 512); hız 96k → 103k tok/s. Train ile val arasındaki fark ≤0.03, yani ezberleme yok; dropout sadece öğrenmeyi yavaşlatıyordu. Kaynaklar: nanoGPT ("pretraining için 0 iyi"), Liu ve ark. 2025 "Drop Dropout on Single-Epoch LM Pretraining". Risk: ~1.87 epoch; val loss izlenecek, en iyi checkpoint zaten ayrı kaydediliyor. |
| 27 Ekim | block_size 256 kaldı (512 yerine) | Belgelerin %81'i 256 token'dan uzun, yani denemeye değerdi. 98M, dropout 0, aynı token bütçesi (64×256 = 32×512): 6000 adımda eğitim val'i eşitlendi (3.8730'a karşı 3.8735; adım 1000'de fark 0.23'tü). Aynı 512'lik pencerelerde adil ölçümde 256 hâlâ önde: TR 3.858'e karşı 3.926, EN 3.785'e karşı 3.839. 512 pencerenin ikinci yarısında bile (2 kat bağlam) geride. 512 ayrıca %5 yavaş. |
| 28 Ekim | Arke-0 ile karşılaştırma aynı metinde, karakter başına | Token başına loss tokenizer'a bağlı; farklı val setlerindeki karakter başına sayılar da metin türüne bağlı (Arke-0 kendi wiki val'inde 1.100, web'de 1.23–1.27). Adil set: hem Arke-0'ın val'ine (mix.txt'nin son %10'u) hem arke0_web'in val'ine (%1 hash) düşen 32 wiki makalesi. Arke-0'ın eski val'i kullanılamazdı: makalelerin %99'u arke0_web'in eğitim verisinde. Yöntemin kontrolü: Arke-0 bu sette 1.110, README'deki kendi sonucu 1.100. |
| 28 Ekim | Örnekler `arke0_web_last.pt` (122k) ile | Tam val'de "best" (118k) checkpoint'ten biraz daha iyi (3.1606'ya karşı 3.1636). |
