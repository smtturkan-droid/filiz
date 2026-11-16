"""filiz-sozluk: Filiz dil modellerinin Türkçe kök-ek sözlüğü (8.192 token). Apache 2.0.

    from filiz_sozluk import FilizSozluk
    s = FilizSozluk()
    ids = s.encode("Evlerimizdekiler geldi.")
    s.decode(ids)                       # -> "Evlerimizdekiler geldi."
    s.tokens("Evlerimizdekiler geldi.")  # -> ['<|B|>', '▁ev', 'lerimiz', 'deki', 'ler', '▁gel', 'di', '.']
    s.sohbet([{"role": "user", "content": "Türkiye'nin başkenti neresidir?"}])   # üretim için istem kimlikleri
"""
from ._sozluk import OZEL, Sozluk8k

__version__ = "0.1.0"
ROL = {"system": "<|sistem|>", "user": "<|kullanici|>", "assistant": "<|filiz|>"}


class FilizSozluk:
    def __init__(self, yol=None):
        self._s = Sozluk8k.yukle(yol)
        self.vocab = self._s.vocab
        self.eos_token_id = self._s.kimlik["<|son|>"]
        self.pad_token_id = self.eos_token_id

    def __len__(self):
        return len(self.vocab)

    def encode(self, metin):
        return self._s.kodla(metin)

    def tokens(self, metin):
        return self._s.parcala(metin)

    def decode(self, ids, ozel_atla=False):
        ids = [int(i) for i in ids]
        if ozel_atla:
            ozel = {self._s.kimlik[t] for t in OZEL if t not in ("<|B|>", "<|BB|>")}
            ids = [i for i in ids if i not in ozel]
        return self._s.coz(ids)

    def token_to_id(self, t):
        return self._s.kimlik.get(t)

    def sohbet(self, mesajlar, uretim_istemi=True):
        """[{"role": system|user|assistant, "content": ...}] -> kimlikler. Araçlar sistem metninde: 'Araçlar: [JSON]'."""
        m = "".join(f"{ROL[x['role']]}\n{x['content']}<|son|>\n" for x in mesajlar)
        return self.encode(m + ("<|filiz|>\n" if uretim_istemi else ""))
