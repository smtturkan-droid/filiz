"""Filiz-118M kullanım örneği / usage example.
  pip install transformers torch git+https://github.com/smtturkan-droid/filiz
  python ornek.py
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from filiz_sozluk import FilizSozluk

sys.stdout.reconfigure(encoding="utf-8")
yol = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent)
model = AutoModelForCausalLM.from_pretrained(yol).eval()
s = FilizSozluk()


def sor(mesajlar, n=60):
    x = torch.tensor([s.sohbet(mesajlar)])
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=n, do_sample=False, eos_token_id=s.eos_token_id,
                           begin_suppress_tokens=[s.token_to_id("<|dusun|>")],
                           pad_token_id=s.pad_token_id)
    return s.decode(y[0, x.shape[1]:].tolist()).replace("<|son|>", "").strip()


# 1) Soru-cevap
print(sor([{"role": "user", "content": "Türkiye'nin başkenti neresidir?"}]))

# 2) Verilen metinden cevap (RAG okuyucu)
metin = "Saat kulesi 1901'de yapıldı. Kulenin yüksekliği 27 metredir ve saati hâlâ çalışır."
print(sor([{"role": "user", "content": f"{metin}\nKulenin yüksekliği kaç metredir?"}]))

# 3) Araç çağrısı: araçlar sistem metninde tanımlanır, model <|arama|>ad {JSON} üretir
araclar = [{"name": "alarm_set", "parameters": ["time", "date"]}, {"name": "weather_query", "parameters": ["place_name", "date"]},
           {"name": "play_music", "parameters": ["artist_name", "song_name"]}]
print(sor([{"role": "system", "content": "Araçlar: " + json.dumps(araclar, ensure_ascii=False, separators=(",", ":"))},
           {"role": "user", "content": "yarın sabah yedide beni uyandır"}]))
