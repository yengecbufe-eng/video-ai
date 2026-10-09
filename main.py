#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VidyoAI — sadece vidyo yapan yapay zeka.
LLM yok, API yok, torch yok. Sadece NumPy + OpenCV.

Nasıl çalışır:
  1) Kod çalışınca BINLERCE KELİMELİK bir eğitim metni (corpus) üretilir:
     binlerce Türkçe sahne cümlesi + doğru etiketleri. Her çalıştırmada
     aynı tohumla üretilir (deterministik) ama kodun İÇİNDE DEĞİLDİR,
     çalışma anında oluşur.
  2) "Anlama Ağı" (saf NumPy MLP) bu metinle eğitilir:
     cümledeki kelimeler -> hangi sahnelerin geçtiği tahmin edilir.
  3) Sizin promptunuz eğitilmiş ağ ile çözümlenir (sözlük araması DEĞİL).
  4) Prompt'un hash'inden tohum türetilir, her vidyoya özel küçük bir
     "Stil Ağı" eğitilir ve sahnenin renklerini modüle eder.
  5) Prosedürel sahne motoru her kareyi NumPy ile boyar -> MP4.

Kullanım:
  python vidyo_ai.py "gün batımı deniz kuşlar"
  python vidyo_ai.py "gece yağmur şehir" --saniye 8 --fps 30 --cikis klip.mp4
  python vidyo_ai.py "uzay yıldız nebula" --cozunurluk 1280x720
  python vidyo_ai.py "..." --ornek 5000 --epoch 300   (eğitim ayarları)
"""

import argparse
import difflib
import hashlib
import math
import os
import shutil
import subprocess
import sys
import time
import wave

import numpy as np
import cv2

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_VAR = True
except Exception:
    PIL_VAR = False

# Windows konsolu UTF-8 sorunlarını önle
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ----------------------------------------------------------------------------
# 1) KELİME DAĞARCIĞI — eğitim metni bu kelimelerden ve şablonlardan üretilir
# ----------------------------------------------------------------------------

KELIMELER = {
    "deniz":     ["deniz", "okyanus", "dalga", "dalgalar", "sahil", "kumsal",
                  "plaj", "kıyı", "köpük", "liman", "ada"],
    "dag":       ["dağ", "dağlar", "tepe", "tepeler", "zirve", "vadi",
                  "kayalık", "uçurum", "yamaç"],
    "sehir":     ["şehir", "kent", "bina", "binalar", "gökdelen", "sokak",
                  "cadde", "metropol", "köprü"],
    "orman":     ["orman", "ağaç", "ağaçlar", "çam", "yaprak", "yeşillik",
                  "koru", "gölge"],
    "uzay":      ["uzay", "galaksi", "nebula", "yıldız", "yıldızlar",
                  "gezegen", "kozmos", "meteor", "yıldız tozu"],
    "yagmur":    ["yağmur", "yağmurlu", "sağanak", "çisenti", "fırtına",
                  "şimşek", "gök gürültüsü", "ıslak"],
    "kar":       ["kar", "karlı", "buz", "buzul", "kış", "dondurucu",
                  "pamuk"],
    "ates":      ["ateş", "alev", "alevler", "yangın", "lav", "volkan",
                  "kor", "duman", "köz"],
    "kuslar":    ["kuş", "kuşlar", "martı", "martılar", "kanat", "uçan",
                  "uçuşan", "sürü"],
    "gunbatimi": ["günbatımı", "alacakaranlık", "akşam", "turuncu",
                  "kızıllık", "günbatısı", "son ışıklar"],
    "gece":      ["gece", "karanlık", "ay", "ayışığı", "geceyarısı",
                  "gece vakti", "loş"],
    "neon":      ["neon", "siberpunk", "synthwave", "retro", "vaporwave",
                  "neonlu", "mor ışıklar"],
    "bulut":     ["bulut", "bulutlar", "bulutlu", "pus", "sis", "sisli",
                  "puslu"],
    "balik":     ["balık", "balıklar", "denizaltı", "sualtı", "mercan",
                  "yosun", "sürü"],
}

# etiketsiz (nötr) kelimeler — ağ bunların hiçbir sahne anlamı olmadığını öğrenir
DOLGU_KELIMELER = [
    "bir", "ve", "ile", "çok", "gibi", "için", "bugün", "şimdi", "burada",
    "orada", "manzara", "manzarası", "görünüm", "görünümü", "sahne",
    "sahnesi", "huzurlu", "sakin", "büyük", "küçük", "geniş", "derin",
    "uzak", "yakın", "gizemli", "harika", "muhteşem", "sessiz", "sıcak",
    "yumuşak", "hafif", "yavaş", "uzun", "kısa", "yeni", "eski", "güzel",
    "ilginç", "gerçek", "hikaye", "görsel", "kamera", "çekim", "vidyo",
    "parça", "an", "anlık", "duruş", "hareket", "ruh", "renk", "ışık",
    "gölge", "hatıra", "rüya", "masal", "şiir",
]

SABLONLAR = [
    "{yer} manzarası",
    "bugün {yer} görmek istiyorum",
    "{zaman} vakti {yer}",
    "{yer} üzerinde {nesne}",
    "{yer} ve {nesne} bir arada",
    "{hava} bir {zaman}, {yer} görünümü",
    "çok {hava} bir {zaman}, {yer} var",
    "{yer}, {zaman} ve {nesne}",
    "{yer} içinde {nesne} hikayesi",
    "bir {zaman} {yer} çekimi, {nesne} ile",
    "{hava} {yer} sahnesi",
    "{nesne} olan {yer} manzarası",
]

ZAMAN_ETIKETLERI = ["gunbatimi", "gece"]  # bu etiketler "zaman" yuvağını doldurur

ETIKETLER = list(KELIMELER.keys())  # 14 etiket


def tokenize(metin):
    return [t for t in metin.lower().replace(",", " ").replace(".", " ").split() if t]


# ----------------------------------------------------------------------------
# 2) EĞİTİM METNİ ÜRETİCİ — kod çalışınca binlerce kelimelik corpus oluşur
# ----------------------------------------------------------------------------

def egitim_metni_uret(n_ornek=5000, seed=42):
    """Binlerce kelimelik eğitim metni üretir: (cümleler, X, Y).
    X: kelime dağarcığı çok-sıcak (multi-hot) vektörleri
    Y: sahne etiketleri (0/1)"""
    rng = np.random.default_rng(seed)

    # kelime -> dizin haritası
    sozluk = []
    for kelimeler in KELIMELER.values():
        for k in kelimeler:
            sozluk.append(k)
    sozluk += DOLGU_KELIMELER
    sozluk = sorted(set(sozluk))
    sozluk_index = {k: i for i, k in enumerate(sozluk)}

    cümleler, X, Y = [], [], []
    for _ in range(n_ornek):
        k = int(rng.integers(1, 4))  # 1-3 pozitif sahne
        poz = list(rng.choice(ETIKETLER, size=k, replace=False))
        sablon = SABLONLAR[int(rng.integers(0, len(SABLONLAR)))]
        # zaman yuvağı SADECE cümlede gerçekten zaman sahnesi varsa zaman kelimesi alır;
        # yoksa nötr zaman kelimesi kullanılır (etiket çelişkisi olmasın)
        zaman_sahnesi = [e for e in poz if e in ZAMAN_ETIKETLERI]
        zaman_pool = ([w for e in zaman_sahnesi for w in KELIMELER[e]]
                      if zaman_sahnesi else ["gündüz", "öğle", "öğleden sonra", "sabah"])
        yer_pool = [w for e in poz for w in KELIMELER[e]]
        nesne_pool = yer_pool + DOLGU_KELIMELER
        hava_pool = DOLGU_KELIMELER

        cümle = sablon
        for yuva, havuz in (("{yer}", yer_pool), ("{nesne}", nesne_pool),
                            ("{zaman}", zaman_pool), ("{hava}", hava_pool)):
            # şablonda zaman yuvağı var ama cümle zaman sahnesi içermiyorsa nötr zaman kullan
            if yuva == "{zaman}" and not zaman_sahnesi:
                cümle = cümle.replace(yuva, zaman_pool[int(rng.integers(0, len(zaman_pool)))])
                continue
            if yuva in cümle:
                cümle = cümle.replace(yuva, havuz[int(rng.integers(0, len(havuz)))])

        # hedef vektör
        y = np.zeros(len(ETIKETLER), dtype=np.float32)
        for e in poz:
            y[ETIKETLER.index(e)] = 1.0

        # çok-sıcak giriş
        x = np.zeros(len(sozluk), dtype=np.float32)
        for tok in tokenize(cümle):
            i = sozluk_index.get(tok)
            if i is not None:
                x[i] = 1.0

        cümleler.append(cümle)
        X.append(x)
        Y.append(y)

    return cümleler, sozluk, np.array(X), np.array(Y)


# ----------------------------------------------------------------------------
# 3) ANLAMA AĞI — saf NumPy MLP (torch YOK), eğitim metniyle eğitilir
# ----------------------------------------------------------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class AnlamaAgi:
    """Kelime torbası -> sahne etiketleri. 2 katmanlı MLP, elle gradyan inişi."""

    def __init__(self, n_giris, gizli=128, n_cikis=len(ETIKETLER), seed=7):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.4, size=(n_giris, gizli))
        self.b1 = np.zeros(gizli)
        self.W2 = rng.normal(0, 0.4, size=(gizli, n_cikis))
        self.b2 = np.zeros(n_cikis)

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = np.tanh(Z1)
        Z2 = A1 @ self.W2 + self.b2
        return sigmoid(Z2), A1

    def train(self, X, Y, epoch=400, lr=0.5, yaz=lambda s: None):
        n = len(X)
        for ep in range(1, epoch + 1):
            _lr = lr * (1.0 - 0.6 * ep / epoch)  # öğrenme oranını yavaşça düşür
            P, A1 = self.forward(X)
            hata = P - Y
            kayip = -np.mean(Y * np.log(P + 1e-9) + (1 - Y) * np.log(1 - P + 1e-9))
            dZ2 = hata / n
            gW2 = A1.T @ dZ2
            gb2 = dZ2.sum(axis=0)
            dA1 = dZ2 @ self.W2.T
            dZ1 = dA1 * (1 - A1 ** 2)
            gW1 = X.T @ dZ1
            gb1 = dZ1.sum(axis=0)
            self.W2 -= _lr * gW2
            self.b2 -= _lr * gb2
            self.W1 -= _lr * gW1
            self.b1 -= _lr * gb1
            if ep % max(1, epoch // 10) == 0:
                dogru = ((P > 0.5) == (Y > 0.5)).mean()
                yaz(f"  epoch {ep:4d}/{epoch}  kayıp={kayip:.4f}  doğruluk=%{dogru * 100:.1f}")
        return kayip

    def tahmin(self, x):
        P, _ = self.forward(x[None, :])
        return P[0]

    def tahmin_cümle(self, cümle, sozluk):
        """Cümleyi kelime torbasına çevir (bilinmeyen kelimeleri en yakın
        dağarcık kelimesine eşle) ve tahmin üret."""
        sozluk_dict = {w: i for i, w in enumerate(sozluk)}
        x = np.zeros(len(sozluk), dtype=np.float32)
        for tok in tokenize(cümle):
            if tok in sozluk_dict:
                x[sozluk_dict[tok]] = 1.0
                continue
            # morfoloji toleransı: "denizde" -> "deniz"
            eslesen = [w for w in sozluk if len(w) >= 4 and (tok.startswith(w) or w.startswith(tok))]
            if not eslesen:
                eslesen = difflib.get_close_matches(tok, sozluk, n=1, cutoff=0.75)
            for w in eslesen[:1]:
                x[sozluk.index(w)] = 1.0
        return self.tahmin(x)


# ----------------------------------------------------------------------------
# 4) SAHNE KURULUMU — tahmin edilen etiketleri çizim parametrelerine çevir
# ----------------------------------------------------------------------------

PALETTES = {
    "gece":      {"sky_top": (8, 10, 34),    "sky_mid": (16, 24, 64),   "sky_horizon": (40, 50, 100)},
    "gunbatimi": {"sky_top": (56, 30, 110),  "sky_mid": (235, 110, 60), "sky_horizon": (255, 190, 120)},
    "gunduz":    {"sky_top": (90, 160, 240), "sky_mid": (150, 200, 250), "sky_horizon": (210, 235, 255)},
    "uzay":      {"sky_top": (4, 4, 12),     "sky_mid": (10, 8, 30),    "sky_horizon": (30, 15, 55)},
    "orman":     {"sky_top": (140, 190, 200), "sky_mid": (170, 210, 190), "sky_horizon": (210, 230, 200)},
    "ates":      {"sky_top": (20, 8, 8),     "sky_mid": (90, 30, 10),   "sky_horizon": (180, 70, 15)},
    "neon":      {"sky_top": (15, 5, 35),    "sky_mid": (60, 10, 80),   "sky_horizon": (130, 20, 110)},
    "firtina":   {"sky_top": (35, 40, 50),   "sky_mid": (60, 70, 85),   "sky_horizon": (95, 105, 120)},
    "kis":       {"sky_top": (150, 180, 220), "sky_mid": (190, 210, 235), "sky_horizon": (235, 240, 250)},
}


def spec_yap(prompt, etiket_seti):
    if "uzay" in etiket_seti:
        pal = "uzay"
    elif "gece" in etiket_seti and not ({"deniz", "sehir"} & etiket_seti):
        pal = "gece"
    elif "ates" in etiket_seti:
        pal = "ates"
    elif "neon" in etiket_seti:
        pal = "neon"
    elif "yagmur" in etiket_seti:
        pal = "firtina"
    elif "kar" in etiket_seti:
        pal = "kis"
    elif "orman" in etiket_seti:
        pal = "orman"
    elif "gunbatimi" in etiket_seti:
        pal = "gunbatimi"
    elif "deniz" in etiket_seti:
        pal = "gunduz"
    else:
        pal = "gunduz"

    seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12], 16)
    return {
        "prompt": prompt,
        "seed": seed,
        "palette": PALETTES[pal],
        "palette_key": pal,
        "deniz": "deniz" in etiket_seti,
        "dag": "dag" in etiket_seti or pal in ("orman", "gunbatimi"),
        "sehir": "sehir" in etiket_seti,
        "yildiz": pal in ("uzay", "gece", "neon") or "uzay" in etiket_seti,
        "ay": "uzay" in etiket_seti or "gece" in etiket_seti,
        "yagmur": "yagmur" in etiket_seti,
        "kar": "kar" in etiket_seti,
        "ates": "ates" in etiket_seti,
        "kuslar": "kuslar" in etiket_seti,
        "bulut": "bulut" in etiket_seti or pal in ("gunduz", "gunbatimi", "orman", "kis"),
        "nebula": "uzay" in etiket_seti,
        "balik": "balik" in etiket_seti,
    }


# ----------------------------------------------------------------------------
# 5) STİL AĞI — saf NumPy, her vidyoya özel renk imzası
# ----------------------------------------------------------------------------

class StilAgi:
    """(x, y, t) -> renk modülasyonu üreten minik MLP (Fourier öznitelikli)."""

    def __init__(self, seed: int, n_feat: int = 24, hidden: int = 40):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n_feat = n_feat
        self.W = rng.normal(0, 1.0, size=(n_feat, 3)) * np.array([3.0, 3.0, 2.0])
        self.b = rng.uniform(0, 2 * math.pi, size=n_feat)
        self.W1 = rng.normal(0, 0.5, size=(2 * n_feat, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.3, size=(hidden, 3))
        self.b2 = np.zeros(3)

    def _feats(self, pts):
        ang = pts @ self.W.T + self.b
        return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)

    def forward(self, pts):
        F = self._feats(pts)
        H = np.tanh(F @ self.W1 + self.b1)
        return np.tanh(H @ self.W2 + self.b2)

    def train(self, steps: int = 250, lr: float = 0.05):
        rng = self.rng
        pts = rng.uniform(0, 2 * math.pi, size=(512, 3))
        target = np.tanh(pts @ rng.normal(0, 0.6, size=(3, 3)))
        for _ in range(steps):
            out = self.forward(pts)
            err = out - target
            d_out = 2.0 * err * (1.0 - out ** 2)
            H = np.tanh(self._feats(pts) @ self.W1 + self.b1)
            dH = (d_out @ self.W2.T) * (1.0 - H ** 2)
            F = self._feats(pts)
            self.W2 -= lr * (H.T @ d_out / len(pts))
            self.b2 -= lr * d_out.mean(axis=0)
            self.W1 -= lr * (F.T @ dH / len(pts))
            self.b1 -= lr * dH.mean(axis=0)


# ----------------------------------------------------------------------------
# 6) GÜRÜLTÜ (value noise + fbm)
# ----------------------------------------------------------------------------

def _hash01(ix, iy, seed):
    return np.sin(ix * 127.1 + iy * 311.7 + seed * 74.7) * 43758.5453 % 1.0


def value_noise(x, y, seed):
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    n00 = _hash01(ix, iy, seed)
    n10 = _hash01(ix + 1, iy, seed)
    n01 = _hash01(ix, iy + 1, seed)
    n11 = _hash01(ix + 1, iy + 1, seed)
    return (n00 * (1 - fx) + n10 * fx) * (1 - fy) + (n01 * (1 - fx) + n11 * fx) * fy


def fbm(x, y, seed, octaves=4):
    total, amp, freq, norm = 0.0, 0.5, 1.0, 0.0
    for o in range(octaves):
        total += amp * value_noise(x * freq, y * freq, seed + o * 17)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return total / norm


# ----------------------------------------------------------------------------
# 7) SAHNE MOTORU
# ----------------------------------------------------------------------------

class VidyoMotoru:
    def __init__(self, w, h, spec: dict, fps: int):
        self.w, self.h, self.spec, self.fps = w, h, spec, fps
        self.rng = np.random.default_rng(spec["seed"])
        self.pal = spec["palette"]

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self.xx, self.yy = xx / w, yy / h
        self.xpix, self.ypix = xx, yy

        self.horizon = 0.62 if spec["deniz"] or spec["dag"] or spec["sehir"] else 1.2

        d = np.sqrt((self.xx - 0.5) ** 2 + (self.yy - 0.5) ** 2)
        self.vignette = np.clip(1.15 - 0.55 * d ** 2, 0.55, 1.0)[..., None].astype(np.float32)

        n_stars = 500 if spec["yildiz"] else 0
        self.stars = np.stack([
            self.rng.uniform(0, 1, n_stars),
            self.rng.uniform(0, 1, n_stars) * min(self.horizon, 1.0),
            self.rng.uniform(0.5, 4.0, n_stars),
            self.rng.uniform(1, 3, n_stars),
        ], axis=1) if n_stars else np.zeros((0, 4))

        self.skyline = self._build_skyline()
        self.ridge1 = fbm(np.linspace(0, 6, w), np.zeros(w), spec["seed"] + 5, 4)
        self.ridge2 = fbm(np.linspace(0, 9, w), np.zeros(w), spec["seed"] + 9, 4)

        self.ag = StilAgi(spec["seed"])
        self.ag.train()

        self._init_particles()

    def _build_skyline(self):
        rng = self.rng
        n = rng.integers(14, 22)
        edges = np.sort(rng.uniform(0, 1, n * 2)).reshape(n, 2)
        heights = rng.uniform(0.18, 0.55, n)
        return list(zip(edges, heights))

    def _init_particles(self):
        s = self.spec
        rng = self.rng
        n = 700 if s["yagmur"] else 500 if s["kar"] else 400 if s["ates"] else 0
        if s["balik"] and not s["ates"]:
            n = 220
        self.p_pos = rng.uniform(0, 1, (n, 2)) if n else np.zeros((0, 2))
        self.p_vel = np.zeros((0, 2))
        self.p_phase = rng.uniform(0, 6.28, n) if n else np.zeros(0)
        if s["yagmur"]:
            self.p_vel = np.stack([rng.uniform(-0.05, -0.02, n), rng.uniform(0.5, 0.9, n)], axis=1)
        elif s["kar"]:
            self.p_vel = np.stack([rng.uniform(-0.05, 0.05, n), rng.uniform(0.05, 0.15, n)], axis=1)
        elif s["ates"]:
            self.p_vel = np.stack([rng.uniform(-0.03, 0.03, n), rng.uniform(-0.25, -0.08, n)], axis=1)
        elif s["balik"]:
            self.p_vel = np.stack([rng.uniform(0.02, 0.08, n) * rng.choice([-1, 1], n), rng.uniform(-0.01, 0.01, n)], axis=1)

    # ---- katmanlar ----
    def _sky(self, t):
        h, w = self.h, self.w
        top = np.array(self.pal["sky_top"], np.float32)
        mid = np.array(self.pal["sky_mid"], np.float32)
        hor = np.array(self.pal["sky_horizon"], np.float32)

        g = np.clip(self.ypix / max(self.horizon, 0.35), 0, 1)
        sky = np.where(g[..., None] < 0.55,
                       top + (mid - top) * (g[..., None] / 0.55),
                       mid + (hor - mid) * ((g[..., None] - 0.55) / 0.45))

        qw, qh = w // 6, h // 6
        qx, qy = np.meshgrid(np.linspace(0, 2 * math.pi, qw), np.linspace(0, 2 * math.pi, qh))
        pts = np.stack([qx.ravel(), qy.ravel(), np.full(qx.size, t * 2 * math.pi)], axis=1).astype(np.float32)
        mod = self.ag.forward(pts).reshape(qh, qw, 3)
        mod = cv2.resize(mod, (w, h), interpolation=cv2.INTER_LINEAR) * 22.0
        sky = sky + mod * (0.4 + 0.6 * g[..., None]) * 0.5
        return np.clip(sky, 0, 255).astype(np.float32)

    def _celestial(self, sky, t):
        s = self.spec
        w, h = self.w, self.h
        if s["ates"]:
            return sky
        if s["ay"] or s["palette_key"] == "uzay":
            cx, cy, r, col = 0.75, 0.18 + 0.03 * math.sin(t * 2), 0.045, np.array([230, 230, 240], np.float32)
        elif s["palette_key"] in ("gunbatimi", "gece"):
            cx, cy = 0.5 + 0.1 * math.sin(t * 0.7), self.horizon - 0.08 + 0.06 * math.cos(t * 0.9)
            r, col = 0.06, np.array([255, 220, 160], np.float32)
        else:
            cx, cy, r, col = 0.72, 0.16, 0.05, np.array([255, 250, 220], np.float32)

        dx = (self.xx - cx) * (w / h)
        dy = self.yy - cy
        dist = np.sqrt(dx * dx + dy * dy)
        glow = np.exp(-dist * 4.0) * 0.85
        disc = np.clip((r - dist) / 0.01, 0, 1)
        if s["ay"]:
            crater = value_noise(self.xx * 40, self.yy * 40, self.spec["seed"]) * 0.15
            disc = disc * (1 - crater)
        sky = sky + glow[..., None] * col * 0.6 + disc[..., None] * col
        return sky

    def _stars_layer(self, sky, t):
        if not len(self.stars):
            return sky
        sx, sy, ph, br = self.stars.T
        tw = 0.6 + 0.4 * np.sin(t * self.fps * 0.35 + ph)
        px = (sx * self.w).astype(int)
        py = (sy * self.h).astype(int)
        ok = (px >= 0) & (px < self.w) & (py >= 0) & (py < self.h)
        sky[py[ok], px[ok]] = np.clip(sky[py[ok], px[ok]] + br[ok, None] * tw[ok, None] * 60, 0, 255)
        return sky

    def _nebula(self, sky, t):
        if not self.spec["nebula"]:
            return sky
        qw, qh = self.w // 5, self.h // 5
        qx, qy = np.meshgrid(np.linspace(0, 5, qw), np.linspace(0, 5, qh))
        n = fbm(qx + t * 0.15, qy, self.spec["seed"] + 77, 4)
        n = cv2.resize(n, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        neb = np.clip(n - 0.45, 0, 1) * 2.2
        col = np.array([120, 40, 160], np.float32)
        return sky + neb[..., None] * col * 0.35

    def _clouds(self, sky, t):
        if not self.spec["bulut"]:
            return sky
        qw, qh = self.w // 5, self.h // 5
        qx, qy = np.meshgrid(np.linspace(0, 4, qw), np.linspace(0, 4, qh))
        n = fbm(qx + t * 0.35, qy * 1.6, self.spec["seed"] + 31, 4)
        n = cv2.resize(n, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        m = np.clip((n - 0.52) * 4.0, 0, 1)
        band = np.clip(1.0 - self.ypix / max(self.horizon, 0.5), 0, 1) ** 1.5
        alpha = m * band * 0.75
        white = np.array([245, 245, 250], np.float32)
        if self.spec["palette_key"] in ("gunbatimi", "ates", "neon"):
            white = np.array([255, 170, 140], np.float32)
        return sky * (1 - alpha[..., None]) + white * alpha[..., None]

    def _mountains(self, sky, t):
        if not self.spec["dag"]:
            return sky
        for i, (ridge, depth) in enumerate([(self.ridge2, 0.18), (self.ridge1, 0.28)]):
            base = self.horizon - 0.02 + i * 0.02
            prof = base - ridge * depth
            mask = self.yy >= prof[None, :]
            shade = np.array([60 + i * 25, 55 + i * 20, 75 + i * 25], np.float32)
            col = shade + (np.array(self.pal["sky_horizon"]) - shade) * (0.35 - i * 0.15)
            sky[mask] = sky[mask] * (1 - 0.85) + col * 0.85
        return sky

    def _city(self, sky, t):
        if not self.spec["sehir"]:
            return sky
        h, w = self.h, self.w
        night = self.spec["palette_key"] in ("gece", "neon", "firtina") or self.spec["yildiz"]
        base_y = self.horizon if self.spec["deniz"] else 0.92
        for (x0, x1), bh in self.skyline:
            xa, xb = int(x0 * w), int(x1 * w)
            y0 = int((base_y - bh * 0.35) * h)
            col = np.array([25, 22, 35], np.float32) if night else np.array([70, 75, 90], np.float32)
            sky[y0:int(base_y * h) if self.spec["deniz"] else h, xa:xb] = col
            if night:
                rng = np.random.default_rng(int(x0 * 10000) + self.spec["seed"] % 1000)
                for wy in range(y0 + 4, int((base_y - 0.02) * h), 8):
                    for wx in range(xa + 3, xb - 3, 7):
                        if rng.random() < 0.28 + 0.15 * math.sin(t * 2 + wx):
                            c = np.array([255, 230, 120], np.float32) if rng.random() < 0.8 else np.array([120, 220, 255], np.float32)
                            sky[wy:wy + 2, wx:wx + 2] = c
        return sky

    def _sea(self, sky, t):
        if not self.spec["deniz"]:
            return sky
        h, w = self.h, self.w
        hy = int(self.horizon * h)
        deep = np.array([10, 40, 70], np.float32)
        if self.spec["palette_key"] in ("gunbatimi", "gece", "neon", "firtina", "ates"):
            deep = np.array([25, 25, 60], np.float32)
        g = np.clip((self.ypix - self.horizon) / max(1e-3, 1 - self.horizon), 0, 1)
        water = deep[None, None, :] + (np.array([120, 190, 220], np.float32) - deep) * (1 - g[..., None]) * 0.25
        wave = np.sin(self.xx * 90 + t * 6 + np.sin(self.ypix * 30 + t * 3) * 2) * 0.5 + 0.5
        wave *= np.clip(1 - g, 0, 1) ** 0.6
        water += wave[..., None] * np.array([180, 200, 220], np.float32) * 0.18
        if self.spec["palette_key"] in ("gunbatimi", "gunduz", "gece") or self.spec["ay"]:
            dx = np.abs(self.xx - 0.55) * (w / h)
            shim = np.exp(-dx * 60) * np.clip(1 - g, 0, 1) * (0.5 + 0.5 * np.sin(self.ypix * 120 + t * 9))
            water += shim[..., None] * np.array([255, 200, 120], np.float32) * 0.5
        sky[hy:, :] = water[hy:, :]
        return sky

    def _underwater(self, sky, t):
        if not self.spec["balik"]:
            return sky
        sky = sky * np.array([0.75, 0.9, 1.05], np.float32)
        sky += np.array([0, 25, 45], np.float32) * (1 - self.yy[..., None])
        beam = np.clip(np.sin(self.xx * 8 + t) * 0.5 + 0.5, 0, 1) ** 6
        sky += beam[..., None] * np.array([60, 90, 110], np.float32) * (1 - self.yy[..., None]) * 0.4
        return sky

    def _fish(self, img, t, dt):
        if not self.spec["balik"] or not len(self.p_pos):
            return img
        self.p_pos += self.p_vel * dt
        self.p_pos[:, 0] %= 1.0
        self.p_pos[:, 1] = np.clip(self.p_pos[:, 1] + np.sin(t * 2 + self.p_phase) * 0.002, 0.1, 0.95)
        for (x, y) in self.p_pos:
            px, py = int(x * self.w), int(y * self.h)
            d = 1 if self.p_vel[np.argmin(np.abs(self.p_pos[:, 0] - x)), 0] > 0 else -1
            cv2.ellipse(img, (px, py), (7, 3), 0 if d > 0 else 180, 0, 360, (250, 200, 120), -1)
            cv2.ellipse(img, (px - 7 * d, py), (3, 3), 0, 0, 360, (240, 170, 100), -1)
        return img

    def _particles(self, img, t, dt):
        s = self.spec
        n = len(self.p_pos)
        if not n:
            return img
        self.p_pos += self.p_vel * dt
        if s["yagmur"]:
            self.p_pos[:, 1] %= 1.1
            self.p_pos[:, 0] = (self.p_pos[:, 0] + self.p_vel[:, 0] * dt) % 1.0
            for x, y in self.p_pos:
                px, py = int(x * self.w), int(y * self.h)
                cv2.line(img, (px, py), (px - 3, py + int(self.h * 0.02)), (200, 210, 230), 1)
        elif s["kar"]:
            self.p_pos[:, 1] %= 1.05
            self.p_pos[:, 0] = ((self.p_pos[:, 0] + self.p_vel[:, 0] * dt + 0.005 * np.sin(t * 3 + self.p_phase)) % 1.0)
            for (x, y), ph in zip(self.p_pos, self.p_phase):
                r = 1 + int(2 * (0.5 + 0.5 * math.sin(ph)))
                cv2.circle(img, (int(x * self.w), int(y * self.h)), r, (250, 250, 255), -1)
        elif s["ates"]:
            self.p_pos[:, 1] = (self.p_pos[:, 1] + self.p_vel[:, 1] * dt) % 1.0
            self.p_pos[:, 0] += self.p_vel[:, 0] * dt
            life = 1.0 - self.p_pos[:, 1]
            for (x, y), lf in zip(self.p_pos, life):
                px, py = int(x * self.w), int(int(0.15 * self.h) + y * self.h * 0.9)
                if lf < 0.2:
                    col = (60, 40, 40)
                elif lf < 0.5:
                    col = (40, 90, 230)
                else:
                    col = (60, 180, 255)
                r = max(1, int(5 * lf))
                cv2.circle(img, (px, py), r, col, -1)
        return img

    def _birds(self, img, t):
        if not self.spec["kuslar"]:
            return img
        h, w = self.h, self.w
        for i in range(7):
            bx = ((t * 0.09 + i * 0.13) % 1.15 - 0.07) * w
            by = (0.12 + 0.08 * math.sin(t * 1.2 + i) + i * 0.025) * h
            flap = math.sin(t * 9 + i * 1.7) * 4
            p1 = (int(bx - 7), int(by - flap))
            p2 = (int(bx), int(by))
            p3 = (int(bx + 7), int(by - flap))
            cv2.polylines(img, [np.array([p1, p2, p3])], False, (20, 20, 25), 2)
        return img

    def _lightning(self, img, t):
        if not self.spec["yagmur"]:
            return img
        if self.rng.random() < 0.02:
            x = int(self.rng.uniform(0.1, 0.9) * self.w)
            pts = [(x, 0)]
            y = 0
            while y < self.h * 0.55:
                x += int(self.rng.integers(-30, 31))
                y += int(self.rng.integers(20, 60))
                pts.append((x, y))
            cv2.polylines(img, [np.array(pts)], False, (255, 255, 255), 2)
            img += 60
        return img

    def frame(self, t, dt):
        sky = self._sky(t)
        sky = self._nebula(sky, t)
        sky = self._celestial(sky, t)
        sky = self._stars_layer(sky, t)
        sky = self._clouds(sky, t)
        sky = self._mountains(sky, t)
        sky = self._city(sky, t)
        sky = self._sea(sky, t)
        sky = self._underwater(sky, t)

        img = np.clip(sky, 0, 255).astype(np.uint8)
        img = self._birds(img, t)
        img = self._fish(img, t, dt)
        img = self._particles(img, t, dt)
        img = self._lightning(img, t)

        grain = (self.rng.normal(0, 4.0, (self.h, self.w, 1))).astype(np.float32)
        img = np.clip(img.astype(np.float32) + grain, 0, 255) * self.vignette
        return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# 8) SES MOTORU — prosedürel ambiyans (saf NumPy -> WAV)
# ----------------------------------------------------------------------------

AKORLAR = {
    "gunduz":    [261.63, 329.63, 392.00],   # C majör — parlak
    "gunbatimi": [220.00, 261.63, 329.63],   # A minör — sıcak hüzün
    "gece":      [174.61, 207.65, 261.63],   # F minör — karanlık
    "uzay":      [130.81, 155.56, 196.00],   # C minör — derin boşluk
    "orman":     [293.66, 349.23, 440.00],   # D majör — taze
    "ates":      [98.00, 146.83, 196.00],    # düşük drone — tehlike
    "neon":      [220.00, 277.18, 329.63],   # A majör7 — retro
    "firtina":   [155.56, 185.00, 233.08],   # G minör — gergin
    "kis":       [246.94, 293.66, 369.99],   # B minör — serin
}


def _yavas_gurultu(rng, sure, sr, kadans=4.0):
    """Düşük frekanslı pürüzsüz gürültü (rüzgar/deniz tabanı)."""
    m = int(sure * kadans) + 2
    nz = rng.normal(0, 1, m)
    return np.interp(np.arange(int(sure * sr)) / sr,
                     np.linspace(0, sure, m), nz)


def ses_mixi_uret(spec, sure, sr=44100):
    """Sahneye göre prosedürel ambiyans müziği üretir, mono float dizi döner."""
    rng = np.random.default_rng(spec["seed"] + 999)
    n = int(sure * sr)
    t = np.arange(n) / sr
    mix = np.zeros(n)

    # ---- akor yastığı (pad) ----
    pal_k = spec["palette_key"]
    for i, f in enumerate(AKORLAR[pal_k]):
        faz = rng.uniform(0, 6.28)
        lfo = 0.8 + 0.2 * np.sin(2 * math.pi * 0.07 * t + faz)
        mix += 0.10 * lfo * np.sin(2 * math.pi * f * t + faz)
        mix += 0.05 * lfo * np.sin(2 * math.pi * f / 2 * t)   # sub-oktav

    # ---- deniz dalgaları ----
    if spec["deniz"]:
        dalga = _yavas_gurultu(rng, sure, sr, kadans=3.0)
        env = (0.5 + 0.5 * np.sin(2 * math.pi * 0.13 * t + rng.uniform(0, 6.28))) ** 2
        mix += 0.30 * dalga * env

    # ---- yağmur + gök gürültüsü ----
    if spec["yagmur"]:
        beyaz = rng.normal(0, 1, n)
        hizli = beyaz - np.convolve(beyaz, np.ones(48) / 48, mode="same")
        mix += 0.22 * hizli
        for _ in range(max(1, int(sure / 3))):
            basla = int(rng.uniform(0, max(1, sure - 1.5)) * sr)
            uzun = int(1.5 * sr)
            gurleme = _yavas_gurultu(rng, 1.5, sr, kadans=8.0)
            zarf = np.exp(-np.linspace(0, 6, uzun))
            son = min(n, basla + uzun)
            mix[basla:son] += 0.5 * gurleme[: son - basla] * zarf[: son - basla]

    # ---- rüzgar ----
    if spec["kar"] or spec["palette_key"] in ("orman", "kis"):
        ruzgar = _yavas_gurultu(rng, sure, sr, kadans=2.0)
        env = 0.5 + 0.5 * np.sin(2 * math.pi * 0.09 * t + rng.uniform(0, 6.28))
        mix += 0.20 * ruzgar * env

    # ---- ateş çıtırtısı ----
    if spec["ates"]:
        mix += 0.12 * _yavas_gurultu(rng, sure, sr, kadans=30.0)
        cakma = (rng.random(n) < 0.0012).astype(np.float32)
        zarf = np.exp(-np.arange(600) / 120.0)
        cakma = np.convolve(cakma, zarf, mode="same")
        mix += 0.35 * cakma * np.sin(2 * math.pi * 90 * t)

    # ---- kuş cıvıltıları ----
    if spec["kuslar"] and not spec["yagmur"]:
        for _ in range(max(2, int(sure))):
            basla = int(rng.uniform(0.2, max(0.3, sure - 0.4)) * sr)
            uz = int(0.18 * sr)
            f0, f1 = rng.uniform(2400, 3200), rng.uniform(3400, 4200)
            fr = np.linspace(f0, f1, uz)
            civ = np.sin(2 * math.pi * np.cumsum(fr) / sr)
            zarf = np.sin(np.linspace(0, math.pi, uz)) ** 2
            son = min(n, basla + uz)
            mix[basla:son] += 0.10 * civ[: son - basla] * zarf[: son - basla]

    # ---- normalizasyon + giriş/çıkış geçişi ----
    tepe = np.max(np.abs(mix)) or 1.0
    mix *= 0.85 / tepe
    fade = int(0.8 * sr)
    mix[:fade] *= np.linspace(0, 1, fade)
    mix[-fade:] *= np.linspace(1, 0, fade)

    return mix


def wav_yaz(mix, yol, sr=44100):
    """Mono float karışımı stereo WAV olarak yazar."""
    mix = np.asarray(mix, dtype=np.float32)
    kaydir = sr // 12
    sag = np.concatenate([np.zeros(kaydir), mix[:-kaydir]]) * 0.95
    stereo = np.stack([mix, sag], axis=1)
    pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
    with wave.open(yol, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def wav_yukle(yol, hedef_sr=44100):
    """WAV dosyasını mono float olarak yükler, gerekirse yeniden örneklemez."""
    with wave.open(yol) as wf:
        sr = wf.getframerate()
        kanal = wf.getnchannels()
        veri = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if kanal == 2:
        veri = veri.reshape(-1, 2).mean(axis=1)
    if sr != hedef_sr:
        t0 = np.arange(len(veri)) / sr
        t1 = np.arange(int(len(veri) * hedef_sr / sr)) / hedef_sr
        veri = np.interp(t1, t0, veri)
    return veri


def konustur(metin, wav_yol, hiz=0):
    """Windows'un kendi konuşma motoru (System.Speech) ile metni sese çevirir.
    Tamamen çevrimdışı — API değil. Konuşma süresini (sn) döner, başarısızsa None.
    Türkçe ses yüklüyse onu seçer, değilse varsayılan sesi kullanır."""
    metin_esc = metin.replace("'", "''")
    ps = f'''Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$tr = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Culture.Name -like 'tr*' }} | Select-Object -First 1
if ($tr) {{ $s.SelectVoice($tr.VoiceInfo.Name) }}
$s.Rate = {int(hiz)}
$s.SetOutputToWaveFile("{wav_yol}")
$s.Speak('{metin_esc}')
$s.Dispose()
'''
    ps1 = wav_yol + ".ps1"
    with open(ps1, "w", encoding="utf-8-sig") as f:
        f.write(ps)
    try:
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1],
                       check=True, capture_output=True, timeout=120)
        with wave.open(wav_yol) as wf:
            sure = wf.getnframes() / wf.getframerate()
        return sure if sure > 0.2 else None
    except Exception:
        return None
    finally:
        try:
            os.remove(ps1)
        except Exception:
            pass


def sesi_mp4e_gom(mp4, wav):
    """ffmpeg varsa sesi MP4 içine gömer. Başarılıysa True döner."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    tmp = mp4 + ".sesli.mp4"
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", mp4, "-i", wav,
             "-c:v", "copy", "-c:a", "aac", "-strict", "-2", "-b:a", "192k", "-shortest", tmp],
            check=True)
        os.replace(tmp, mp4)
        os.remove(wav)
        return True
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


# ----------------------------------------------------------------------------
# 9) ALTYAZI MOTORU — prompt sinematik yazı olarak vidyoya basılır
# ----------------------------------------------------------------------------

def altyazi_hazirla(metin, w, h):
    """Prompt metnini RGBA katman olarak hazırlar (PIL varsa Türkçe harflerle)."""
    katman = np.zeros((h, w, 4), dtype=np.uint8)

    if PIL_VAR:
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ciz = ImageDraw.Draw(img)
        boyut = max(16, w // 34)
        font = None
        for yol in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf",
                    r"C:\Windows\Fonts\segoeui.ttf"):
            try:
                font = ImageFont.truetype(yol, boyut)
                break
            except Exception:
                pass
        if font is None:
            font = ImageFont.load_default()

        # uzun metni satırlara böl
        maks_gen = int(w * 0.86)
        satirlar, satir = [], ""
        for kelime in metin.split():
            dene = (satir + " " + kelime).strip()
            if ciz.textbbox((0, 0), dene, font=font)[2] > maks_gen and satir:
                satirlar.append(satir)
                satir = kelime
            else:
                satir = dene
        satirlar.append(satir)

        satir_yuk = boyut + boyut // 3
        toplam = len(satirlar) * satir_yuk
        y0 = h - toplam - int(h * 0.055)
        for i, satir in enumerate(satirlar):
            tw = ciz.textbbox((0, 0), satir, font=font)[2]
            x = (w - tw) // 2
            y = y0 + i * satir_yuk
            ciz.text((x + 2, y + 2), satir, font=font, fill=(0, 0, 0, 200))   # gölge
            ciz.text((x, y), satir, font=font, fill=(255, 255, 255, 235))

        # sağ üst köşe filigran
        kucuk = max(12, w // 60)
        try:
            f2 = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", kucuk)
        except Exception:
            f2 = font
        ciz.text((w - 10 - ciz.textbbox((0, 0), "VidyoAI", font=f2)[2], 8),
                 "VidyoAI", font=f2, fill=(255, 255, 255, 140))
        katman = np.array(img)
    else:
        # PIL yoksa: Türkçe harfleri ASCII'ye çevirip cv2 ile yaz
        tr = str.maketrans("ğğıöçüşİ", "ggioocusI")
        metin = metin.translate(tr)
        boyut = max(0.5, w / 900.0)
        tw, th = cv2.getTextSize(metin, cv2.FONT_HERSHEY_DUPLEX, boyut, 2)[0]
        x, y = (w - tw) // 2, h - int(h * 0.06)
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(bgr, metin, (x + 2, y + 2), cv2.FONT_HERSHEY_DUPLEX,
                    boyut, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(bgr, metin, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                    boyut, (255, 255, 255), 2, cv2.LINE_AA)
        katman[..., :3] = bgr[..., ::-1]          # BGR -> RGB
        katman[..., 3] = bgr.max(axis=2)          # çizili yerler opak
    return katman


def altyazi_uygula(frame, katman, alpha):
    """alpha 0..1 — yazıyı kareye yumuşak geçişle basar."""
    a = (katman[..., 3].astype(np.float32) / 255.0) * alpha
    f = frame.astype(np.float32)
    out = f * (1 - a[..., None]) + katman[..., :3] * a[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# 10) ANA PROGRAM
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="VidyoAI — LLM'siz, API'siz, torch'suz vidyo üretici")
    ap.add_argument("prompt", nargs="?", default="gün batımı deniz kuşlar", help="Türkçe/İngilizce prompt")
    ap.add_argument("--saniye", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--cozunurluk", default="960x540")
    ap.add_argument("--cikis", default="vidyo.mp4")
    ap.add_argument("--ornek", type=int, default=5000, help="eğitim metni örnek (cümle) sayısı")
    ap.add_argument("--epoch", type=int, default=400, help="eğitim turu sayısı")
    ap.add_argument("--sessiz", action="store_true", help="ambiyans sesini kapat")
    ap.add_argument("--altyazisiz", action="store_true", help="altyazıyı kapat")
    ap.add_argument("--konus", default=None, help='vidyoya söyletecek metin, ör: --konus "merhaba dünya"')
    ap.add_argument("--hiz", type=int, default=0, help="konuşma hızı (-10 yavaş ... +10 hızlı)")
    args = ap.parse_args()

    try:
        w, h = map(int, args.cozunurluk.lower().split("x"))
    except ValueError:
        print("Çözünürlük formatı: GENISLIKxYUKSEKLIK, örn 1280x720")
        sys.exit(1)

    # ---- 1. EĞİTİM METNİ (kod çalışınca üretilir) ----
    print("[1/5] Eğitim metni üretiliyor...")
    t0 = time.time()
    cumleler, sozluk, X, Y = egitim_metni_uret(n_ornek=args.ornek)
    toplam_kelime = sum(len(tokenize(c)) for c in cumleler)
    print(f"      {len(cumleler)} cümle üretildi — TOPLAM {toplam_kelime:,} kelime")
    print(f"      kelime dağarcığı: {len(sozluk)} benzersiz kelime")
    ornek = cumleler[int(np.random.default_rng(1).integers(0, len(cumleler)))]
    print(f'      örnek cümle : "{ornek}"')

    # ---- 2. EĞİTİM ----
    print("[2/5] Anlama Ağı eğitiliyor...")
    ag = AnlamaAgi(len(sozluk))
    ag.train(X, Y, epoch=args.epoch, yaz=lambda s: print("      " + s))
    P, _ = ag.forward(X)
    dogruluk = ((P > 0.5) == (Y > 0.5)).mean()
    tam = ((P > 0.5).astype(int) == Y.astype(int)).all(axis=1).mean()
    print(f"      eğitim bitti ({time.time() - t0:.1f} sn) — hücre doğruluğu %{dogruluk * 100:.2f}, tam örnek doğruluğu %{tam * 100:.1f}")

    # ---- 3. PROMPT ANLAMA ----
    print("[3/5] Prompt çözümleniyor...")
    olasiliklar = ag.tahmin_cümle(args.prompt, sozluk)
    etiket_seti = {ETIKETLER[i] for i, p in enumerate(olasiliklar) if p > 0.5}
    print("      tahmin olasılıkları:")
    for i, p in enumerate(olasiliklar):
        bar = "█" * int(p * 24)
        print(f"        {ETIKETLER[i]:<10} %{p * 100:5.1f} {bar}")
    if not etiket_seti:
        print("      (belirsiz prompt -> sakin gündüz sahnesi)")
    spec = spec_yap(args.prompt, etiket_seti)
    print(f"      seçilen sahne: {sorted(etiket_seti) or ['gunduz (sakin)']}, palet: {spec['palette_key']}")

    # ---- 3.5 KONUŞMA ----
    sure = args.saniye
    konus_wav = args.cikis.rsplit(".", 1)[0] + "_konusma.wav"
    konus_hazir = False
    if args.konus:
        print("[3.5] Konuşma sentezleniyor (Windows TTS, çevrimdışı)...")
        ksure = konustur(args.konus, konus_wav, hiz=args.hiz)
        if ksure:
            konus_hazir = True
            sure = max(args.saniye, ksure + 1.0)
            print(f'      konuşma hazır: {ksure:.1f} sn -> vidyo süresi {sure:.1f} sn')
        else:
            print("      Windows konuşma motoru kullanılamadı, sessiz devam")

    # ---- 4. VİDYO ÜRETİMİ ----
    print("[4/5] Vidyo üretiliyor...")
    motor = VidyoMotoru(w, h, spec, args.fps)
    total = int(sure * args.fps)   # konuşma varsa vidyo otomatik uzar
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.cikis, fourcc, args.fps, (w, h))
    if not vw.isOpened():
        print("VideoWriter açılamadı."); sys.exit(1)

    print(f"      {w}x{h} @ {args.fps}fps, {sure:.1f} sn, {total} kare")

    # altyazı katmanı (konuşma metni varsa onu göster)
    altyazi_metni = args.konus if args.konus else args.prompt
    katman = None if args.altyazisiz else altyazi_hazirla(altyazi_metni, w, h)

    t1 = time.time()
    for i in range(total):
        t = i / args.fps
        frame = motor.frame(t, 1.0 / args.fps)
        if katman is not None:
            giris = min(1.0, t / 1.2)
            cikis = min(1.0, max(0.0, (sure - t)) / 1.2)
            frame = altyazi_uygula(frame, katman, min(giris, cikis))
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if i % max(1, total // 10) == 0:
            print(f"      kare {i}/{total}  (%{100 * i // total})")
    vw.release()

    # ---- 5. SES ----
    if args.sessiz and not konus_hazir:
        print("[5/5] Ses kapalı (--sessiz)")
    else:
        print("[5/5] Ses karıştırılıyor...")
        sr = 44100
        n = int(sure * sr)
        mix = np.zeros(n, dtype=np.float32)
        if not args.sessiz:
            mix += ses_mixi_uret(spec, sure, sr) * (0.4 if konus_hazir else 1.0)
        if konus_hazir:
            kon = wav_yukle(konus_wav, sr)
            gecikme = int(0.3 * sr)
            son = min(n, gecikme + len(kon))
            mix[gecikme:son] += kon[: son - gecikme] * 0.95
            os.remove(konus_wav)
            print(f'      konuşma eklendi: "{args.konus}"')
        tepe = float(np.max(np.abs(mix)) or 1.0)
        if tepe > 0.9:
            mix *= 0.9 / tepe
        wav_yol = args.cikis.rsplit(".", 1)[0] + "_ses.wav"
        wav_yaz(mix, wav_yol, sr)
        if sesi_mp4e_gom(args.cikis, wav_yol):
            print("      ses MP4 içine gömüldü 🎵")
        else:
            print(f"      ses MP4'e gömülemedi — ayrı dosya: {wav_yol}")

    print(f"Bitti! Eğitim+üretim toplam {time.time() - t0:.1f} sn -> {args.cikis}")


if __name__ == "__main__":
    main()
