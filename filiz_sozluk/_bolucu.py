"""Türkçe kök | ek bölücü (Filiz 8k sözlüğünün çekirdeği). Ekler kalıptan ve dil bilgisi sırasıyla üretilir; kökler veriden."""
import json
import gzip
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

VERI = Path(__file__).resolve().parent / "veri"
UNLU = set("aeıioöuüâîû")
KALIN = set("aıouâû")
YUV = set("ouöüû")
SERT = set("çfhkpsşt")
RMIN, FMIN, EMIN, BETA, DELTA, GAMMA, GAMMA_K = 30, 3, 50, 0.02, 0.01, 0.2, 0.5
IR_TEK = {"al", "bil", "bul", "dur", "gel", "gör", "kal", "ol", "öl", "san", "var", "ver", "vur"}
KAPALI = set("""ve ile ise için gibi göre diye kadar karşı rağmen dolayı ötürü üzere beri dek değin sonra önce ancak ama fakat
lakin çünkü eğer ki de da mi mı mu mü bile hem ya veya yahut ne değil bana sana ben sen biz siz örneğin yani hani işte yalnız
sadece belki sanki hatta artık zaten yine gene henüz hiç hep daha çok az en pek iki üç dört beş yedi sekiz dokuz on yüz
bin milyon şey nasıl neden niye niçin hangi kim kimse herkes hepsi biri birisi böyle şöyle öyle şimdi bugün yarın dün""".split())
KOK_OLAMAZ = set("""ve ile ise de da ki mi mı mu mü ama fakat lakin çünkü eğer bile hem ya veya yahut ne daha en çok az pek hiç
hep için göre diye kadar ancak yani hani işte sanki belki hatta zaten dek beni seni""".split())   # bağlaç / edat: kök olarak da kullanılmaz


def son_unlu(s):
    for c in reversed(s):
        if c in UNLU:
            return c
    return ""


def cevir(h):
    return {"a": "e", "ı": "i", "o": "ö", "u": "ü", "â": "e", "û": "ü"}.get(h, h)


# ---------------------------------------------------------------- 1. ek kalıpları
# ad -> (kalıp, sınıf). sınıf: A = addan ad yapım (kesilir), V = çatı: edilgen / ettirgen (kesilir: ed·il, araştır·ıl),
#   D = fiilden ad (BİRLEŞİK kalır: bilgi, bilim, durum, uzatma; kullanıcı 2026-11-10), L = addan fiil -lA/-lAn (TDK tanımına göre),
#   K = karşılıklı (sözlükselleşebilir: konuş), C = çekim
EK = {
    # addan ad yapım
    "lI": ("lI", "A"), "lIK": ("lIK", "A"), "sIz": ("sIz", "A"), "CI": ("CI", "A"), "sAl": ("sAl", "A"), "DAş": ("DAş", "A"),
    "CIK": ("CIK", "A"), "msI": ("(I)msI", "A"), "lAş": ("lAş", "A"),
    # çoğul, iyelik, hal
    "lAr": ("lAr", "C"), "P1": ("(I)m", "C"), "P2": ("(I)n", "C"), "P3": ("(s)I", "C"), "P4": ("(I)mIz", "C"), "P5": ("(I)nIz", "C"),
    "BEL": ("(y)I", "C"), "YON": ("(y)A", "C"), "BUL": ("DA", "C"), "AYR": ("DAn", "C"), "ILG": ("(n)In", "C"), "VAS": ("(y)lA", "C"),
    "ESIT": ("CA", "C"), "nBEL": ("nI", "C"), "nYON": ("nA", "C"), "nBUL": ("nDA", "C"), "nAYR": ("nDAn", "C"), "ki": ("ki", "C"),
    # ek-fiil, kişi
    "DIr": ("DIr", "C"), "kDI": ("(y)DI", "C"), "kMIS": ("(y)mIş", "C"), "kSA": ("(y)sA", "C"), "ken": ("(y)ken", "C"),
    "K1": ("(y)Im", "C"), "K2": ("sIn", "C"), "K4": ("(y)Iz", "C"), "K5": ("sInIz", "C"), "K6": ("lAr", "C"),
    "k1": ("m", "C"), "k2": ("n", "C"), "k4": ("k", "C"), "k5": ("nIz", "C"), "lIm": ("lIm", "C"), "sInlAr": ("sInlAr", "C"),
    "E2": ("(y)In", "C"), "E5": ("(y)InIz", "C"),
    # fiil: çatı
    "ETT": ("DIr", "V"), "ETTt": ("t", "V"), "EDL": ("EDL", "V"), "IS": ("(I)ş", "K"),
    # yeterlik, olumsuz, kip / zaman
    "Abil": ("(y)Abil", "C"), "OLZ": ("mA", "C"), "AmA": ("(y)AmA", "C"), "Iyor": ("(I)yor", "C"), "mIyor": ("mIyor", "C"),
    "AmIyor": ("(y)AmIyor", "C"), "DI": ("DI", "C"), "mIs": ("mIş", "C"), "AcAK": ("(y)AcAK", "C"), "GEN": ("GEN", "C"),
    "z": ("z", "C"), "mAlI": ("mAlI", "C"), "sA": ("sA", "C"), "IST": ("(y)A", "C"),
    # adlaştırma (fiilden ad: her zaman kesilir) ve sıfat / zarf-fiil (çekim sayılır)
    "mA": ("mA", "D"), "mAK": ("mAK", "C"), "Is": ("(y)Iş", "D"), "IcI": ("(y)IcI", "D"), "Im": ("(I)m", "D"), "GI": ("GI", "D"),
    # addan fiil: -lA / -lAn (TDK tanımında ad kökü geçiyorsa kesilir: hesap·la; geçmiyorsa kök: başla, kullan)
    "lA": ("lA", "L"), "lAn": ("lAn", "L"), "lIyor": ("lIyor", "L"),   # amaç·lıyor (la + ıyor ünlü düşmesi)
    "An": ("(y)An", "C"), "DIK": ("DIK", "C"), "AcAKs": ("(y)AcAK", "C"), "AsI": ("(y)AsI", "C"),
    "ArAk": ("(y)ArAk", "C"), "Ip": ("(y)Ip", "C"), "IncA": ("(y)IncA", "C"), "AlI": ("(y)AlI", "C"), "mAdAn": ("mAdAn", "C"),
    "mAksIzIn": ("mAksIzIn", "C"), "DIkcA": ("DIkçA", "C"),
}

# ---------------------------------------------------------------- 2. sıra (durum makinesi)
KOP = ["DIr", "kDI", "kMIS", "kSA", "ken", "K1", "K2", "K4", "K5"]
HAL = ["BEL", "YON", "BUL", "AYR", "ILG", "VAS", "ESIT"]
NHAL = ["nBEL", "nYON", "nBUL", "nAYR", "ILG", "VAS"]


def _hal(liste):
    return [(h, "HALKI" if h in ("BUL", "nBUL", "ILG") else "HAL") for h in liste]


KOPG = [(k, {"DIr": "DIRS", "kDI": "GEC", "kMIS": "MIS", "kSA": "GEC"}.get(k, "SON")) for k in KOP]
ADLAS = [("mA", "AD"), ("mAK", "AD"), ("Is", "AD"), ("IcI", "AD"), ("Im", "AD"), ("GI", "AD"), ("An", "AD"), ("DIK", "AD"),
         ("AcAKs", "AD"), ("AsI", "AD"), ("ArAk", "SON"), ("Ip", "SON"), ("IncA", "SON"), ("AlI", "SON"), ("mAdAn", "SON"),
         ("mAksIzIn", "SON"), ("DIkcA", "SON")]
GECIS = {
    "AD": [("lI", "AD"), ("lIK", "AD"), ("sIz", "AD"), ("CI", "AD"), ("sAl", "AD"), ("DAş", "AD"), ("CIK", "AD"), ("msI", "AD"),
           ("lAş", "FIIL"), ("lA", "FIIL"), ("lAn", "FIIL"), ("lIyor", "T"), ("ki", "KI"), ("lAr", "COG"), ("P3", "IYE3"), ("P1", "IYE"), ("P2", "IYE"), ("P4", "IYE"), ("P5", "IYE")]
          + _hal(HAL) + KOPG,
    "COG": [("P3", "IYE3"), ("P1", "IYE"), ("P2", "IYE"), ("P4", "IYE"), ("P5", "IYE")] + _hal(HAL) + KOPG,
    "IYE": _hal(HAL) + KOPG,
    "IYE3": _hal(NHAL) + KOPG,
    "HALKI": [("ki", "KI")] + KOPG,
    "HAL": KOPG,
    "KI": [("lAr", "COG")] + _hal(NHAL) + KOPG,
    "DIRS": [("K6", "SON")],
    "FIIL": [("ETT", "FIIL"), ("ETTt", "FIIL"), ("EDL", "FIIL"), ("IS", "FIIL"), ("Abil", "FK"), ("OLZ", "OLZ"), ("AmA", "OLZ"),
             ("mIyor", "T"), ("AmIyor", "T")],
    "FK": [("Iyor", "T"), ("mIyor", "T"), ("AmIyor", "T"), ("OLZ", "OLZ"), ("AmA", "OLZ"), ("DI", "GEC"), ("mIs", "MIS"),
           ("AcAK", "T"), ("GEN", "T"), ("GEN", "AD"), ("mAlI", "T"), ("sA", "GEC"), ("IST", "IST"), ("E2", "SON"), ("E5", "SON"),
           ("K2", "SON"), ("sInlAr", "SON")] + ADLAS,
    "OLZ": [("DI", "GEC"), ("mIs", "MIS"), ("AcAK", "T"), ("z", "T"), ("z", "AD"), ("mAlI", "T"), ("sA", "GEC"), ("IST", "IST"),
            ("k1", "SON"), ("K4", "SON"), ("K2", "SON"), ("E2", "SON"), ("E5", "SON"), ("mA", "AD"), ("mAK", "AD"), ("Is", "AD"),
            ("An", "AD"), ("DIK", "AD"), ("AcAKs", "AD"), ("ArAk", "SON"), ("Ip", "SON"), ("IncA", "SON"), ("AlI", "SON")],
    "T": [("K1", "SON"), ("K2", "SON"), ("K4", "SON"), ("K5", "SON"), ("K6", "TL"), ("kDI", "GEC"), ("kMIS", "MIS"),
          ("kSA", "GEC"), ("DIr", "DIRS"), ("ken", "SON")],
    "TL": [("kDI", "SON"), ("kMIS", "SON"), ("kSA", "SON"), ("DIr", "SON")],
    "MIS": [("K1", "SON"), ("K2", "SON"), ("K4", "SON"), ("K5", "SON"), ("K6", "TL"), ("kDI", "GEC"), ("kSA", "GEC"),
            ("DIr", "DIRS"), ("ken", "SON")],
    "GEC": [("k1", "SON"), ("k2", "SON"), ("k4", "SON"), ("k5", "SON"), ("K6", "SON"), ("kSA", "GEC2"), ("kDI", "GEC2")],
    "GEC2": [("k1", "SON"), ("k2", "SON"), ("k4", "SON"), ("k5", "SON"), ("K6", "SON")],
    "IST": [("K1", "SON"), ("K2", "SON"), ("lIm", "SON"), ("K5", "SON"), ("K6", "SON"), ("kDI", "SON")],
    "SON": [],
}
GECIS["FIIL"] += GECIS["FK"]
SINIF = {k: v[1] for k, v in EK.items()}


def _birim(kalip):
    out, i = [], 0
    while i < len(kalip):
        if kalip[i] == "(":
            j = kalip.index(")", i)
            out.append(("?", kalip[i + 1:j]))
            i = j + 1
        else:
            out.append(("", kalip[i]))
            i += 1
    return out


BIRIM = {k: _birim(v[0]) for k, v in EK.items()}


def _unlu(c, h):
    if c == "A":
        return "a" if h in KALIN else "e"
    if h in KALIN:
        return "u" if h in YUV else "ı"
    return "ü" if h in YUV else "i"


def gercekle(ad, onceki, h, genis):
    """ek ad -> [(yüzey, yeni_h, sonu_ğ)] ; onceki = bağlamın son harfi, h = uyum ünlüsü, genis = 'Ar' / 'Ir' (fiil gövdesi)"""
    unlu_sonra = onceki in UNLU
    if ad == "EDL":
        s = "n" if unlu_sonra else ("In" if onceki == "l" else "Il")
        birim = [("", c) for c in s]
    elif ad == "GEN":
        birim = [("", "r")] if unlu_sonra else [("", genis[0]), ("", "r")]
    elif ad == "ETTt":
        if not (unlu_sonra or onceki in "lr"):
            return []
        birim = [("", "t")]
    else:
        birim = BIRIM[ad]
    s, prev = [], onceki
    for opt, c in birim:
        if opt:
            if c in "yns":
                if not unlu_sonra:
                    continue
            elif unlu_sonra:
                continue
        if c in "AI":
            c = _unlu(c, h)
        elif c == "D":
            c = "t" if prev in SERT else "d"
        elif c == "C":
            c = "ç" if prev in SERT else "c"
        elif c == "G":
            c = "k" if prev in SERT else "g"
        if c in UNLU:
            h = c
        s.append(c)
        prev = c
    if not s:
        return []
    y = "".join(s)
    if y.endswith("K"):
        return [(y[:-1] + "k", h, False), (y[:-1] + "ğ", h, True)]
    return [(y, h, False)]


@lru_cache(maxsize=3_000_000)
def zincir(durum, kalan, onceki, h, genis):
    """kalan dizgeyi durumdan başlayarak eklere böl -> en az ekli çözüm (tuple ek adları, tuple yüzeyler) ya da None"""
    if not kalan:
        return ((), ())
    en = None
    for ad, sonraki in GECIS[durum]:
        for y, h2, ğ in gercekle(ad, onceki, h, genis):
            if not kalan.startswith(y):
                continue
            geri = kalan[len(y):]
            if ğ and (not geri or geri[0] not in UNLU):
                continue
            g2 = "Ir" if sonraki in ("FIIL", "FK") else genis
            alt = zincir(sonraki, geri, y[-1], h2, g2)
            if alt is not None and (en is None or len(alt[0]) + 1 < len(en[0])):
                en = ((ad,) + alt[0], (y,) + alt[1])
    return en


# ---------------------------------------------------------------- 3. kökler
class Bolucu2:
    def __init__(self, ad_kok, fiil_kok, cevrik=frozenset(), kapali=KAPALI, ozel=frozenset(), ikiz_red=frozenset(), guc=None, korunan=frozenset()):
        self.ad, self.fiil, self.cevrik, self.kapali, self.ozel = ad_kok, fiil_kok, cevrik, kapali, ozel
        self.ikiz_red = ikiz_red
        self.korunan = korunan
        self.guc = guc or {}                            # kök gücü: kendi geçişi + mastar + fiil biçimleri
        self.kok = defaultdict(list)                   # yüzey -> [(ana kök, ikiz türü)]
        for r in ad_kok | fiil_kok:
            self.kok[r].append((r, ""))
        for r in ad_kok | fiil_kok:
            for t, tur in ikizler(r, r in fiil_kok):
                if (t, r) not in ikiz_red:
                    self.kok[t].append((r, tur))

    def adaylar(self, w, en_uzun=None, kucuk=True):
        """w için tüm (kök yüzeyi, ana kök, ikiz türü, ekler, yüzeyler, çevrik) çözümleri, uzun kökten kısaya"""
        out = []
        for L in range(min(len(w), en_uzun or len(w)), 0, -1):
            r, z = w[:L], w[L:]
            for ana, tur in self.kok.get(r, ()):
                if kucuk and ana in self.ozel and ana != w:     # özel ad kökü küçük harfli kelimeyi bölmez (aras·ında değil ara·sı-nda)
                    continue
                if not z:
                    if not tur:
                        out.append((r, ana, tur, (), (), False))
                    continue
                if tur and not ikiz_uyar(tur, z):
                    continue
                h0 = son_unlu(ana)
                if not h0:
                    continue
                gen = "r" if ana[-1] in UNLU else ("Ir" if (ana in IR_TEK or sum(c in UNLU for c in ana) > 1) else "Ar")
                if ana in ("o", "bu", "şu") and z[0] == "n":
                    bas, z2, onc = "n", z[1:], "n"
                else:
                    bas, z2, onc = "", z, r[-1]
                for cev in ((False, True) if ana in self.cevrik else (False,)):
                    h = cevir(h0) if cev else h0
                    for durum in (("AD",) if ana in self.ad else ()) + (("FIIL",) if ana in self.fiil else ()):
                        if bas and not z2:
                            continue
                        c = zincir(durum, z2, onc, h, gen) if z2 else ((), ())
                        if c is not None and c[0]:
                            ys = c[1]
                            if bas:
                                ys = (bas + ys[0],) + ys[1:]
                            out.append((r, ana, tur, c[0], ys, cev))
        return out

    def bol(self, w, kucuk=True):
        """-> (kök, [ek yüzeyleri], tür)   tür: kapalı / kök / ekli / zincir / bilinmeyen"""
        if w in self.kapali:
            return w, [], "kapalı"
        if w not in self.ad and saf_zincir(w):          # ad kökü olmayan saf zincir (kesme sonrası: daki, dır, nın, in)
            return "", list(saf_zincir(w)), "zincir"
        a = self.adaylar(w, kucuk=kucuk)
        if a:
            # birden çok geçerli kök: GÜÇLÜ olan (kal·an, kala·n değil; karar, kar·ar değil); eşitlikte uzun kök, az ek, çevriksiz
            if w in self.korunan and any(not x[3] for x in a):
                return w, [], "kök"                     # TDK isim/sıfat ve kendi ad çekimleri var (yazar, adam, veri, alt)
            # kısa kökün çözümü tutulmuş uzun kökün sınırından geçiyorsa uzun kök (ara·ş-tır -> araştır, er·iş -> eriş);
            # sınırlar örtüşmüyorsa (kal·an / kala·n) güçlü kök
            def sinirlar(x):
                out, L = set(), len(x[0])
                for y in x[4]:
                    L += len(y)
                    out.add(L)
                return out
            a = [x for x in a if not any(len(x2[0]) > len(x[0]) and x2[3] and len(x2[0]) in sinirlar(x) for x2 in a)]
            r, ana, tur, ekler, ys, cev = min(a, key=lambda x: (any(SINIF[e] in "DL" for e in x[3]), -self.guc.get(x[1], 0),
                                                                x[5], -len(x[0]), len(x[3])))
            return r, list(ys), ("ekli" if ys else "kök")
        c = saf_zincir(w)
        if c:
            return "", list(c), "zincir"
        return w, [], "bilinmeyen"

    @staticmethod
    def yukle(yol=None):
        d = json.loads(Path(yol or VERI / "bolucu_kok.json").read_text(encoding="utf-8"))
        return Bolucu2(set(d["ad"]), set(d["fiil"]), frozenset(d["cevrik"]), ozel=frozenset(d["ozel"]),
                       ikiz_red=frozenset(tuple(x) for x in d["ikiz_red"]), guc=d["guc"], korunan=frozenset(d["korunan"]))


def saf_zincir(w):
    """kökü olmayan, baştan sona ad ek zinciri (kesme sonrası: nın, lerin, daki) -> yüzeyler ya da None"""
    for onc, h in (("a", "a"), ("e", "e"), ("ı", "ı"), ("i", "i"), ("o", "o"), ("ö", "ö"), ("u", "u"), ("ü", "ü"),
                   ("k", "a"), ("k", "e"), ("t", "a"), ("t", "e"), ("l", "ı"), ("l", "i"), ("l", "u"), ("l", "ü")):
        c = zincir("AD", w, onc, h, "Ir")
        if c is not None and c[0] and all(SINIF[e] == "C" for e in c[0]):   # yalnız çekim ekleri (deniz, taşlar zincir değil)
            return c[1]
    return None


def ikizler(r, fiil):
    """ses değişimli ikizler: (ikiz yüzeyi, tür)"""
    out = []
    son = r[-1]
    cok_heceli = sum(c in UNLU for c in r) >= 2
    if len(r) >= 2:
        if son == "k" and (cok_heceli or r[-2] == "n"):          # tek heceli ad yumuşamaz (katı, otu); renk->rengi istisna
            out.append((r[:-1] + ("g" if r[-2] == "n" else "ğ"), "yum"))
        elif son in "pç" and cok_heceli and not fiil:
            out.append((r[:-1] + {"p": "b", "ç": "c"}[son], "yum"))
        elif son == "t" and (fiil or cok_heceli):               # git->gid, et->ed; umut->umud
            out.append((r[:-1] + "d", "yum"))
    if (not fiil and len(r) >= 4 and r[-1] not in UNLU and r[-2] in "ıiuü" and r[-3] not in UNLU
            and sum(c in UNLU for c in r) == 2):
        out.append((r[:-2] + r[-1], "dus"))
    if fiil and son in "ae" and len(r) >= 2:
        out.append((r[:-1], "fdus"))
        if r in ("de", "ye"):
            out.append((r[0] + "i", "diyi"))
    return out


def ikiz_uyar(tur, z):
    if tur in ("yum", "dus"):
        return z[0] in UNLU
    if tur == "fdus":
        return len(z) >= 4 and z[0] in "ıiuü" and z[1:3] == "yo"
    if tur == "diyi":
        return z[0] == "y"
    return True
