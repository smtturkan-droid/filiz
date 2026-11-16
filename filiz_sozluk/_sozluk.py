"""Filiz 8k kök-ek sözlüğü: metin <-> token kimlikleri."""
import gzip
import json
import math
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

V = Path(__file__).resolve().parent
OUT = V / "sozluk_8k"
import os
BOY, YEDEK, EK_BUTCE = 8192, int(os.environ.get("FILIZ_YEDEK", 1500)), 1000
AD = "sozluk8k.json" if YEDEK == 1500 else f"sozluk8k_y{YEDEK}.json"
OZEL = ['<|son|>', '<|B|>', '<|BB|>', '<|sistem|>', '<|kullanici|>', '<|filiz|>', '<|belge|>', '<|/belge|>', '<|arama|>',
        '<|sonuc|>', '<|/sonuc|>', '<|dusun|>']
HARF = "abcçdefgğhıijklmnoöprsştuüvyzqwxâîû"
PARCA = re.compile(r"<\|/?[a-zA-Z]+\|>|[^\W\d_]+|\d|\s|.", re.UNICODE)
_B, _K = "A-ZÇĞİÖŞÜÂÎÛĢ", "a-zçğıöşüâîûģ"
HARF_PARCA = re.compile(rf"[{_B}]{{2,}}(?=[{_K}])|[{_B}]+(?=[{_B}][{_K}])|[{_B}]?[^\W\d_{_B}]+|[{_B}]+|[^\W\d_]+", re.UNICODE)
ACIK = ("\n", "\t", "(", "[", "“", "‘")               # bunlardan sonra kelime başı ▁ yazılır, coz'da düşer
_KUCUK = str.maketrans("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWXÂÎÛ", "abcçdefgğhıijklmnoöprsştuüvyzqwxâîû")


def kucuk(w):
    return w.translate(_KUCUK).lower()


def dp(ms, birim):
    n = len(ms)
    en = [0] + [10 ** 9] * n
    geri = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - 8), i):
            u = "".join(ms[j:i])
            if i - j == 1 or u in birim:
                if en[j] + 1 < en[i]:
                    en[i], geri[i] = en[j] + 1, j
    out, i = [], n
    while i:
        out.append("".join(ms[geri[i]:i]))
        i = geri[i]
    return out[::-1]


class Sozluk8k:
    def __init__(self, kokler, ekler, yedek_tok, sabit):
        from ._bolucu import Bolucu2
        from tokenizers import Tokenizer
        self.b = Bolucu2.yukle()
        self.kok, self.ek, self.sabit = set(kokler), set(ekler), set(sabit)
        self.yedek = Tokenizer.from_str(yedek_tok)
        v = OZEL + sorted(self.sabit - set(OZEL)) + ["▁" + k for k in sorted(kokler)] + sorted(ekler)
        var = set(v)
        self.vocab = v + sorted(t for t in self.yedek.get_vocab() if t not in var)   # yedek kök / ek ile çakışırsa tek kimlik
        self.kimlik = {t: i for i, t in enumerate(self.vocab)}

    def coz(self, ids, bas=True):
        """kimlikler -> metin. ▁ = kelime başı boşluk (satır / parantez başında ve metin başında düşer), <|B|> baş harf büyük,
        <|BB|> kelimenin tamamı büyük (sonraki ▁'lı tokena dek), <0xXX> bayt dizisi utf-8 olarak"""
        out, bayt = [], bytearray()
        buyuk, bb_basladi = None, False
        ust = lambda x: x.replace("i", "İ").replace("ı", "I").upper()  # noqa: E731

        def bosalt():
            nonlocal buyuk
            if bayt:
                y = bayt.decode("utf-8", "replace")
                if buyuk == "<|BB|>":
                    y = ust(y)
                elif buyuk == "<|B|>":
                    y, buyuk = ust(y[0]) + y[1:], None
                out.append(y)
                bayt.clear()
        for i in ids:
            t = self.vocab[i]
            if t.startswith("<0x") and len(t) == 6:
                bayt.append(int(t[3:5], 16))
                continue
            bosalt()
            if t in ("<|B|>", "<|BB|>"):
                buyuk, bb_basladi = t, False
                continue
            s = t.replace("▁", " ")
            if s.startswith(" ") and ((bas and not out) or (out and (out[-1][-1:] in ACIK or out[-1].endswith("<|son|>")))):
                s = s[1:]
            if buyuk == "<|BB|>":
                if bb_basladi and (t.startswith("▁") or not t[:1].isalpha()):
                    buyuk = None
                else:
                    s, bb_basladi = ust(s), True
            elif buyuk == "<|B|>" and s.strip():
                j = len(s) - len(s.lstrip())
                s, buyuk = s[:j] + ust(s[j]) + s[j + 1:], None
            out.append(s)
        bosalt()
        return "".join(out)

    def kodla(self, metin):
        """metin -> token kimlikleri (uint16'ya sığar)"""
        k = self.kimlik
        return [k[t] for t in self.parcala(metin)]

    @staticmethod
    def yukle(yol=None):
        d = json.loads(Path(yol or Path(__file__).resolve().parent / "veri" / "sozluk8k.json").read_text(encoding="utf-8"))
        return Sozluk8k(d["kokler"], d["ekler"], d["yedek_tokenizer"], d["sabit"])

    def _yedek(self, w, bas):
        """sözlük dışı parça -> BPE yedek parçaları (bas: kelime başı ▁ ile)"""
        enc = self.yedek.encode(("▁" if bas else "") + w).tokens
        if "".join(enc).replace("▁", "") != w:          # BPE alfabesinde olmayan harf (é, ß...) sessizce düşer: bayt yedeği
            return (["▁"] if bas else []) + [f"<0x{b:02X}>" for b in w.encode("utf-8")]
        return enc

    @lru_cache(maxsize=500_000)
    def kelime(self, w):
        """küçük harfli kelime -> token listesi"""
        r, ys, t = self.b.bol(w)
        def ekler(ys):                                   # sözlükte olmayan tek ek biçimi (tabloda görülmemiş) -> yedek parçalar
            return [p for u in dp(ys, self.ek) for p in ([u] if u in self.ek else self._yedek(u, False))]
        if t == "zincir":
            return tuple(ekler(ys))
        if t == "bilinmeyen" or not r:
            return tuple(self._yedek(w, True))
        bas = ["▁" + r] if r in self.kok else self._yedek(r, True)
        return tuple(bas + (ekler(ys) if ys else []))

    def parcala(self, metin):
        """boşluk kuralı (coz ile birebir): ▁ = önünde boşluk var. Metin / satır / parantez başı (ACIK) boşluk sayılır ve coz'da düşer.
        Önünde boşluk olan kelimenin ilk tokenı ▁'lı olmalı (kökü olmayan zincir kelimede ayrı ▁ eklenir: "fark ▁ nedir");
        boşluksuz kelimenin (2001'de, "güçlü) ilk tokenı ▁'sız olmalı (kök ek birimiyse o, değilse yedek parçalar)."""
        out = []
        bosluk, bas = False, True
        for m in PARCA.finditer(metin):
            p = m.group()
            if p in self.sabit and p.startswith("<|"):
                out.append(p)
                bosluk, bas = False, p in ("<|son|>",)
            elif p[0].isalpha():
                for j, q in enumerate(HARF_PARCA.findall(p)):   # karışık harf (ProQuest, kHz, IIIde) boşluksuz parçalara
                    if j == 0 and bas and bosluk:
                        out.append("▁")                      # satır / parantez başında gerçek boşluk ("\n Kara"): coz ilk ▁'ı düşürür
                    if len(q) >= 2 and q.isupper():
                        out.append("<|BB|>")
                    elif q[0].isupper():
                        out.append("<|B|>")
                    k = list(self.kelime(kucuk(q)))
                    if j == 0 and (bosluk or bas):
                        if not k[0].startswith("▁"):
                            k.insert(0, "▁")
                    elif k[0].startswith("▁"):
                        r = k[0][1:]
                        k[0:1] = ([r] if r in self.ek else self._yedek(r, False)) if r else []
                    out.extend(k)
                bosluk = bas = False
            elif p == " ":
                if bosluk:                               # ikinci boşluk: ayrı ▁ tokenı (çift boşluk korunur)
                    out.append("▁")
                bosluk = True
            else:
                if bosluk:
                    out.extend(["▁", "▁"] if bas else ["▁"])   # rakam / noktalama öncesi boşluk ("1 ile 5"); başta coz ilkini düşürür
                if p in self.sabit:
                    out.append(p)
                else:
                    out.extend(f"<0x{b:02X}>" for b in p.encode("utf-8"))
                bosluk, bas = False, p in ACIK
        return out
