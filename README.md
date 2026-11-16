# Filiz

**Az veriyle verimli öğrenen küçük Türkçe dil modelleri.** · [English](#english)

Filiz, sıfırdan eğitilen Türkçe (ve İngilizce) dil modelleri projesidir. Hedef büyük modellerin değil, **küçük, yerel ve
hızlı** çalışan, kendi boyut sınıfının üstündeki modelleri belirli işlerde yakalayan çekirdekler üretmek.

| | |
|---|---|
| 🤗 Model | [smtturkan/filiz-118m](https://huggingface.co/smtturkan/filiz-118m) |
| Kaggle | [samettrkan/filiz-118m](https://www.kaggle.com/models/samettrkan/filiz-118m) |
| Sözlük | bu depo (`filiz_sozluk`) |
| Lisans | Apache 2.0 |

## Filiz-118M

118 milyon parametre, yaklaşık **1,5 milyar token** ile eğitildi. Aynı düzenekte (lm-evaluation-harness 0.4.12, 0-shot):

| Model | Parametre | Eğitim tokenı | XCOPA-tr | TurBLiMP |
|---|---|---|---|---|
| **Filiz-118M** | 118M | ~1,5B | **58,6** | **79,8** |
| ufakzeka-1-base | ~165M | 13,5B | 59,2 | 81,9 |
| turkish-gpt2 | 124M | — | 57,4 | 54,7 |
| SmolLM2-360M | 360M | 4T | 54,2 | 54,8 |
| Qwen3-0.6B (base) | 600M | 36T | 54,0 | 57,1 |

- Türkçe akıl yürütme (XCOPA) ve dil bilgisinde (TurBLiMP) 3-5 kat büyük modelleri geçer; sınıf liderine ~9 kat az veriyle yaklaşır.
- **Araç çağrısı:** MASSIVE tr-TR test bölümünde (eğitimde görülmedi, ince ayarsız) doğru araç + 3-7 çeldirici düzeninde
  niyet doğruluğu **%85,6**; alan değerleri zayıf (niyet + alanlar tam %42,6).
- Temel modeldir: cihaz içi komut / araç çağrısı, verilen metinden cevap (RAG okuyucusu) ve ince ayar başlangıcı için.
  Bilgi doğruluğu düşüktür, hesap yapmaz (hesap aracını çağırır). Ayrıntı: [model kartı](https://huggingface.co/smtturkan/filiz-118m).

## Kurulum ve kullanım

```bash
pip install transformers torch git+https://github.com/smtturkan-droid/filiz
python ornek.py          # soru-cevap, metinden cevap, araç çağrısı
```

```python
import torch
from transformers import AutoModelForCausalLM
from filiz_sozluk import FilizSozluk

model = AutoModelForCausalLM.from_pretrained("smtturkan/filiz-118m").eval()   # standart Qwen3, trust_remote_code yok
s = FilizSozluk()
x = torch.tensor([s.sohbet([{"role": "user", "content": "Türkiye'nin başkenti neresidir?"}])])
y = model.generate(x, max_new_tokens=60, do_sample=False, eos_token_id=s.eos_token_id, pad_token_id=s.pad_token_id,
                   begin_suppress_tokens=[s.token_to_id("<|dusun|>")])
print(s.decode(y[0, x.shape[1]:].tolist()))   # Türkiye'nin başkenti Ankara'dır.
```

## Sözlük: `filiz_sozluk`

8.192 tokenlık Türkçe kök-ek sözlüğü. Kelimeler kök + ek birimlerine bölünür; ekler Türkçe ses uyumu kalıplarından ve
dil bilgisi sırasından (durum makinesi) üretilir, kökler veriden seçilir. Sözlük dışı parçalar için yedek BPE ve bayt
yedeği vardır; her metin kayıpsız kodlanır ve geri çözülür. Büyük harf ayrı işaretle tutulur (`<|B|>`, `<|BB|>`).

```python
from filiz_sozluk import FilizSozluk
s = FilizSozluk()
s.tokens("Evlerimizdekiler geldi.")   # ['<|B|>', '▁ev', 'lerimiz', 'deki', 'ler', '▁gel', 'di', '.']
s.decode(s.encode("Evlerimizdekiler geldi."))   # 'Evlerimizdekiler geldi.'
```

Yalnız `tokenizers` paketine bağlıdır; veri dosyaları JSON'dur (pickle yok).

**Sohbet biçimi:** `<|sistem|>`, `<|kullanici|>`, `<|filiz|>` rolleri, her mesaj `<|son|>` ile biter. Araçlar sistem
metninde `Araçlar: [{"name": ..., "parameters": [...]}]`; model `<|arama|>ad {"alan": "değer"}` üretir, aracın cevabı
`<|sonuc|> … <|/sonuc|>` içinde verilir.

## İletişim

Sorular, iş birliği ve ticari kullanım için: **Samet Türkan** · [smtturkan@gmail.com](mailto:smtturkan@gmail.com) · GitHub [issues](https://github.com/smtturkan-droid/filiz/issues)

## Yol haritası

118M bir vitrin ve deney çekirdeğidir. Sıradaki adım son mimariyle (16-32k Türkçe + İngilizce sözlük, daha uzun bağlam)
daha büyük bir çekirdek ve ardından 1B amiral model.

---

<a name="english"></a>
## English

**Filiz** is a project of small Turkish (and English) language models trained from scratch, aiming for small, local and
fast models that match larger size classes on specific tasks.

**Filiz-118M** (118M params, ~1.5B training tokens): XCOPA-tr **58.6**, TurBLiMP **79.8** (lm-eval 0.4.12, 0-shot) — ahead of
SmolLM2-360M (54.2 / 54.8) and Qwen3-0.6B base (54.0 / 57.1), close to ufakzeka-1-base (59.2 / 81.9, trained on ~9× more tokens).
Tool calling on the MASSIVE tr-TR test split (unseen, no fine-tuning): **85.6%** intent accuracy with the correct tool plus
3–7 distractors; slot values are the weak point. Base model for on-device command / tool calling, RAG reading and fine-tuning.

This repository holds the **tokenizer** (`filiz_sozluk`, an 8,192-token Turkish root–suffix tokenizer, lossless) and a usage
example. The model is a standard `Qwen3ForCausalLM` on the Hugging Face Hub — no `trust_remote_code` needed.

```bash
pip install transformers torch git+https://github.com/smtturkan-droid/filiz
```

**Contact:** Samet Türkan · [smtturkan@gmail.com](mailto:smtturkan@gmail.com) · [GitHub issues](https://github.com/smtturkan-droid/filiz/issues)

License: Apache 2.0 · © 2026 Samet Türkan
