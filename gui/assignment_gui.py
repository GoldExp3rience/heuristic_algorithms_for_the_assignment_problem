#!/usr/bin/env python3

import ctypes
import json
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from scipy.optimize import linear_sum_assignment

# ── DPI-aware на Windows ──────────────────────────────────────────────────────
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# ── Palette (GitHub-Dark вдохновение) ────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
CARD     = "#1c2128"
BORDER   = "#30363d"
ACCENT   = "#58a6ff"
GREEN    = "#3fb950"
RED      = "#f85149"
TEXT     = "#e6edf3"
TEXT_DIM = "#7d8590"
SEL_BG   = "#1b3458"
SEL_BD   = "#388bfd"
DIM_BG   = "#090c10"
DIM_TXT  = "#2d3339"
HDR_BG   = "#21262d"

STRAT_COLORS = {
    "greedy":        "#ff7b72",
    "chi":           "#79c0ff",
    "hungarian":     "#56d364",
    "hybrid":        "#ffa657",
    "hybrid2":       "#d2a8ff",
    "ensemble":      "#f0883e",
    "ensemble_chi":  "#63e6be",
    "half_hybrid":   "#e8b4f8",
    "half_hybrid2":  "#a5d6a7",
}

STRAT_LABELS = {
    "greedy":        "Жадная",
    "chi":           "χ-стратегия",
    "hungarian":     "Венгерский",
    "hybrid":        "Гибридная-1",
    "hybrid2":       "Гибридная-2",
    "ensemble":      "Ансамбль-B",
    "ensemble_chi":  "Ансамбль-χ",
    "half_hybrid":   "Half-Hybrid",
    "half_hybrid2":  "Half-Hybrid-2",
}

# Размеры ячеек сетки назначений
CELL_W  = 72
CELL_H  = 56
HDR_W   = 44
HDR_H   = 32
CHI_W   = 52   # ширина дополнительной колонки χᵢ (показывается рядом с A)
DOT_D   = 8    # диаметр точки стратегии
DOT_G   = 2    # зазор между точками


# ── Алгоритмы ────────────────────────────────────────────────────────────────

def build_G(C, chi):
    """g̃ᵢⱼ = (1 - χᵢ) * Σₜ≥ⱼ cᵢₜ  (матрица G из статьи [1])"""
    n = C.shape[0]
    G = np.zeros((n, n))
    for i in range(n):
        s = 0.0
        for t in range(n - 1, -1, -1):
            s += C[i, t]
            G[i, t] = (1.0 - chi[i]) * s
    return G


def build_D(C, chi):
    """d̃ᵢⱼ = (1 − χᵢ) · cᵢⱼ — критерий жадного выбора на шаге j."""
    return (1.0 - chi[:, np.newaxis]) * C


def calc_profit(C, chi, sched):
    """Полная прибыль для заданного расписания sched[j] = i."""
    n = C.shape[0]
    base  = float(np.sum(chi[:, None] * C))
    bonus = sum((1 - chi[sched[j]]) * float(np.sum(C[sched[j], j:]))
                for j in range(n) if sched[j] >= 0)
    return base + bonus


# ── Стратегии (A и chi известны полностью) ───────────────────────────────────

def _strat_greedy(C, chi):
    """argmax (1-chi[x])*C[x,j] на каждом шаге."""
    n = C.shape[0]
    used, s = set(), [-1] * n
    for j in range(n):
        av = [x for x in range(n) if x not in used]
        best = max(av, key=lambda x: (1 - chi[x]) * C[x, j])
        s[j] = best
        used.add(best)
    return s


def _strat_chi(C, chi):
    """Сортировка по возрастанию chi. Матрица C не используется."""
    return list(np.argsort(chi))


def _strat_hungarian(C, chi):
    """Оптимальное решение на полной G (пост-фактум эталон)."""
    G = build_G(C, chi)
    ri, ci = linear_sum_assignment(-G)
    s = [-1] * G.shape[0]
    for r, c in zip(ri, ci):
        s[c] = r
    return s


def _strat_hybrid1(C, chi):
    """Чётные шаги: min chi; нечётные: argmax (1-chi[x])*C[x,j]."""
    n = C.shape[0]
    used, s = set(), [-1] * n
    for j in range(n):
        av = [x for x in range(n) if x not in used]
        best = min(av, key=lambda x: chi[x]) if j % 2 == 0 \
               else max(av, key=lambda x: (1 - chi[x]) * C[x, j])
        s[j] = best
        used.add(best)
    return s


def _strat_hybrid2(C, chi):
    """Чётные шаги: argmax (1-chi[x])*C[x,j]; нечётные: min chi."""
    n = C.shape[0]
    used, s = set(), [-1] * n
    for j in range(n):
        av = [x for x in range(n) if x not in used]
        best = max(av, key=lambda x: (1 - chi[x]) * C[x, j]) if j % 2 == 0 \
               else min(av, key=lambda x: chi[x])
        s[j] = best
        used.add(best)
    return s


def _strat_half_hybrid(C, chi):
    """Первые ceil(n/2) шагов: min chi; оставшиеся: argmax (1-chi[i])*C[i][j]."""
    import math
    n = C.shape[0]
    half = math.ceil(n / 2)
    used, s = set(), [-1] * n
    for j in range(n):
        av = [x for x in range(n) if x not in used]
        best = min(av, key=lambda x: chi[x]) if j < half \
               else max(av, key=lambda x: (1 - chi[x]) * C[x, j])
        s[j] = best
        used.add(best)
    return s


def _strat_half_hybrid2(C, chi):
    """Первые ceil(n/2) шагов: argmax (1-chi[i])*C[i][j]; оставшиеся: min chi."""
    import math
    n = C.shape[0]
    half = math.ceil(n / 2)
    used, s = set(), [-1] * n
    for j in range(n):
        av = [x for x in range(n) if x not in used]
        best = max(av, key=lambda x: (1 - chi[x]) * C[x, j]) if j < half \
               else min(av, key=lambda x: chi[x])
        s[j] = best
        used.add(best)
    return s


def _strat_ensemble(C, chi, m=20, seed=None):
    """m агентов, каждый на шаге j выбирает argmax (1-chi[x])*C[x,j]
    из случайной выборки кандидатов. Победитель — по итоговой прибыли.
    seed=None — случайный; seed=int — воспроизводимый."""
    n = C.shape[0]
    ss = max(1, n // 3)
    schedules = [[-1] * n for _ in range(m)]
    used = [set() for _ in range(m)]
    rngs = ([np.random.default_rng(seed + a) for a in range(m)]
            if seed is not None
            else [np.random.default_rng() for _ in range(m)])
    for j in range(n):
        for agent in range(m):
            av    = [x for x in range(n) if x not in used[agent]]
            cands = rngs[agent].choice(av, size=min(ss, len(av)), replace=False)
            best  = int(max(cands, key=lambda x: (1 - chi[x]) * C[x, j]))
            schedules[agent][j] = best
            used[agent].add(best)
    return max(schedules, key=lambda s: calc_profit(C, chi, s))


def _strat_ensemble_chi(C, chi, m=20, seed=None):
    """m агентов выбирают min chi из случайной выборки кандидатов.
    Победитель — по итоговой прибыли.
    seed=None — случайный; seed=int — воспроизводимый."""
    n = len(chi)
    ss = max(1, n // 3)
    schedules = [[-1] * n for _ in range(m)]
    used = [set() for _ in range(m)]
    rngs = ([np.random.default_rng(seed + a) for a in range(m)]
            if seed is not None
            else [np.random.default_rng() for _ in range(m)])
    for j in range(n):
        for agent in range(m):
            av    = [x for x in range(n) if x not in used[agent]]
            cands = rngs[agent].choice(av, size=min(ss, len(av)), replace=False)
            best  = int(min(cands, key=lambda x: chi[x]))
            schedules[agent][j] = best
            used[agent].add(best)
    return max(schedules, key=lambda s: calc_profit(C, chi, s))


ALL_STRATS = {
    "greedy":        _strat_greedy,
    "chi":           _strat_chi,
    "hungarian":     _strat_hungarian,
    "hybrid":        _strat_hybrid1,
    "hybrid2":       _strat_hybrid2,
    "half_hybrid":   _strat_half_hybrid,
    "half_hybrid2":  _strat_half_hybrid2,
    "ensemble":      _strat_ensemble,
    "ensemble_chi":  _strat_ensemble_chi,
}


# ── Цветовые утилиты ─────────────────────────────────────────────────────────

def _hex_lerp(c1, c2, t):
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    r1 = int(c1[1:3], 16)
    g1 = int(c1[3:5], 16)
    b1 = int(c1[5:7], 16)
    r2 = int(c2[1:3], 16)
    g2 = int(c2[3:5], 16)
    b2 = int(c2[5:7], 16)
    r = int(r1 * (1 - t) + r2 * t)
    g = int(g1 * (1 - t) + g2 * t)
    b = int(b1 * (1 - t) + b2 * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _val_color(v, vmin, vmax):
    """Нейтральный серый (низкое) → янтарный → зелёный (высокое).
    Контрастирует с тёмным фоном и явно показывает «выгодные» ячейки."""
    if vmax <= vmin:
        return "#232931"
    t = (v - vmin) / (vmax - vmin)
    if t < 0.5:
        return _hex_lerp("#232931", "#5a3e10", t * 2)   # серый → тёмно-янтарный
    return _hex_lerp("#5a3e10", "#1a5c2a", (t - 0.5) * 2)  # янтарный → тёмно-зелёный


# ── Прокручиваемый фрейм ─────────────────────────────────────────────────────

class _ScrollFrame(tk.Frame):
    """Frame с прокруткой — inner содержит дочерние виджеты.

    scroll_x / scroll_y: включить горизонтальную / вертикальную прокрутку.
    """

    def __init__(self, parent, scroll_x=True, scroll_y=True, **kw):
        bg = kw.get("bg", PANEL)
        super().__init__(parent, bg=bg)
        self._cvs = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.inner = tk.Frame(self._cvs, bg=bg)
        self._cvs.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner)

        if scroll_y:
            self._sb_v = tk.Scrollbar(self, orient=tk.VERTICAL,
                                      command=self._cvs.yview)
            self._cvs.configure(yscrollcommand=self._sb_v.set)
            self._sb_v.pack(side=tk.RIGHT, fill=tk.Y)

        if scroll_x:
            self._sb_h = tk.Scrollbar(self, orient=tk.HORIZONTAL,
                                      command=self._cvs.xview)
            self._cvs.configure(xscrollcommand=self._sb_h.set)
            self._sb_h.pack(side=tk.BOTTOM, fill=tk.X)

        self._cvs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._scroll_y = scroll_y

        # Колёсико мыши: привязываем на весь фрейм и его внутренности
        self._cvs.bind("<Enter>",  self._bind_wheel)
        self._cvs.bind("<Leave>",  self._unbind_wheel)

    def _bind_wheel(self, _e=None):
        self._cvs.bind_all("<MouseWheel>",      self._on_wheel)
        self._cvs.bind_all("<Button-4>",        self._on_wheel)  # Linux scroll up
        self._cvs.bind_all("<Button-5>",        self._on_wheel)  # Linux scroll down

    def _unbind_wheel(self, _e=None):
        self._cvs.unbind_all("<MouseWheel>")
        self._cvs.unbind_all("<Button-4>")
        self._cvs.unbind_all("<Button-5>")

    def _on_wheel(self, e):
        if not self._scroll_y:
            return
        if e.num == 4:
            self._cvs.yview_scroll(-1, "units")
        elif e.num == 5:
            self._cvs.yview_scroll(1, "units")
        else:
            self._cvs.yview_scroll(int(-e.delta / 120), "units")

    def _on_inner(self, _e):
        self._cvs.configure(scrollregion=self._cvs.bbox("all"))


# ── Главное приложение ────────────────────────────────────────────────────────

class AssignmentApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Задача о назначениях — Эвристические методы")
        self.configure(bg=BG)
        self.geometry("1380x840")
        self.minsize(980, 680)

        # ── Переменные состояния ──────────────────────────────────────────
        self.n_var       = tk.IntVar(value=5)
        self.show_var    = tk.StringVar(value="B")   # "A" | "B" | "D"
        self.gen_mode    = tk.StringVar(value="desc")
        self.ensemble_m          = tk.IntVar(value=20)
        self.ensemble_fixed_seed = tk.BooleanVar(value=False)
        self.ensemble_seed       = tk.IntVar(value=42)

        # Диапазоны случайной генерации
        self.a_low   = tk.DoubleVar(value=1.0)
        self.a_high  = tk.DoubleVar(value=10.0)
        self.chi_low  = tk.DoubleVar(value=0.3)
        self.chi_high = tk.DoubleVar(value=0.8)

        self._C   = None    # np.ndarray или None
        self._chi = None
        self._G   = None
        self._D   = None

        # ручное назначение: {slot_j: worker_i}
        self._manual = {}
        # результаты стратегий: {name: {"sched", "profit", "time"}}
        self._strat_res = {}

        self._strat_vars = {}
        for k in ALL_STRATS:
            default = k in ("greedy", "hungarian")
            self._strat_vars[k] = tk.BooleanVar(value=default)

        self._a_entries  = []   # list[list[tk.Entry]]
        self._chi_entries = []  # list[tk.Entry]
        self._hover = None      # (i, j) или None

        # ── Строим интерфейс ──────────────────────────────────────────────
        self._build()
        self._rebuild_inputs(self.n_var.get())
        self._do_random()

    # ────────────────────────────────── Layout ────────────────────────────────

    def _build(self):
        self._build_topbar()

        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BG,
                              sashwidth=5, sashrelief=tk.FLAT,
                              handlesize=0)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        left = tk.Frame(pane, bg=PANEL, width=318)
        left.pack_propagate(False)
        pane.add(left, minsize=260, width=318)

        right = tk.Frame(pane, bg=BG)
        pane.add(right, minsize=520)

        self._build_left(left)
        self._build_right(right)

    # ── Верхняя панель ─────────────────────────────────────────────────────────

    def _build_topbar(self):
        bar = tk.Frame(self, bg=PANEL)
        bar.pack(fill=tk.X, padx=8, pady=(8, 6))

        # ── Строка 1: заголовок + параметры генерации ──────────────────────────
        row1 = tk.Frame(bar, bg=PANEL)
        row1.pack(fill=tk.X, padx=6, pady=(6, 2))

        tk.Label(row1, text="ЗАДАЧА О НАЗНАЧЕНИЯХ",
                 bg=PANEL, fg=ACCENT, font=("Segoe UI", 13, "bold")
                 ).pack(side=tk.LEFT, padx=(2, 16))

        self._vsep(row1)

        # n
        tk.Label(row1, text="n =", bg=PANEL, fg=TEXT_DIM,
                 font=("Segoe UI", 10)).pack(side=tk.LEFT)
        self._n_spin = tk.Spinbox(
            row1, from_=2, to=20, width=3, textvariable=self.n_var,
            bg=CARD, fg=TEXT, insertbackground=TEXT,
            buttonbackground=CARD, relief=tk.FLAT,
            font=("Segoe UI", 12, "bold"),
            command=self._on_n_change)
        self._n_spin.pack(side=tk.LEFT, padx=(4, 12))
        self._n_spin.bind("<Return>", self._on_n_change_event)

        self._vsep(row1)

        # Режим генерации строк C
        tk.Label(row1, text="C:", bg=PANEL, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 2))
        rb_desc = tk.Radiobutton(row1, text="↓ убыв", variable=self.gen_mode, value="desc",
                                  bg=PANEL, fg=TEXT_DIM, selectcolor=CARD,
                                  activebackground=PANEL, activeforeground=TEXT,
                                  font=("Segoe UI", 9))
        rb_desc.pack(side=tk.LEFT, padx=2)
        rb_asc = tk.Radiobutton(row1, text="↑ возр", variable=self.gen_mode, value="asc",
                                 bg=PANEL, fg=TEXT_DIM, selectcolor=CARD,
                                 activebackground=PANEL, activeforeground=TEXT,
                                 font=("Segoe UI", 9))
        rb_asc.pack(side=tk.LEFT, padx=2)
        rb_rand = tk.Radiobutton(row1, text="случайн", variable=self.gen_mode, value="rand",
                                  bg=PANEL, fg=TEXT_DIM, selectcolor=CARD,
                                  activebackground=PANEL, activeforeground=TEXT,
                                  font=("Segoe UI", 9))
        rb_rand.pack(side=tk.LEFT, padx=2)

        self._vsep(row1)

        # Диапазоны случайной генерации
        def _rl(parent, txt):
            tk.Label(parent, text=txt, bg=PANEL, fg=TEXT_DIM,
                     font=("Segoe UI", 8)).pack(side=tk.LEFT)

        _rl(row1, "c ∈ [")
        self._mk_range_spin(row1, self.a_low,  0.01, 99999, 1.0,
                            is_chi=False).pack(side=tk.LEFT)
        _rl(row1, " – ")
        self._mk_range_spin(row1, self.a_high, 0.01, 99999, 1.0,
                            is_chi=False).pack(side=tk.LEFT)
        _rl(row1, "]")

        tk.Label(row1, text="   ", bg=PANEL).pack(side=tk.LEFT)

        _rl(row1, "χ ∈ (")
        self._mk_range_spin(row1, self.chi_low,  0.01, 0.98, 0.05,
                            is_chi=True).pack(side=tk.LEFT)
        _rl(row1, " – ")
        self._mk_range_spin(row1, self.chi_high, 0.02, 0.99, 0.05,
                            is_chi=True).pack(side=tk.LEFT)
        _rl(row1, ")")

        # Тонкий разделитель между строками
        tk.Frame(bar, bg=BORDER, height=1).pack(fill=tk.X, padx=6)

        # ── Строка 2: кнопки действий ──────────────────────────────────────────
        row2 = tk.Frame(bar, bg=PANEL)
        row2.pack(fill=tk.X, padx=6, pady=(2, 6))

        btn_rand = self._mk_btn(row2, "🎲 Случайно", self._do_random, ACCENT)
        btn_rand.pack(side=tk.LEFT, padx=(0, 6))
        btn_open = self._mk_btn(row2, "📂 Открыть", self._load_file, CARD)
        btn_open.pack(side=tk.LEFT, padx=(0, 6))
        btn_save = self._mk_btn(row2, "💾 Сохранить", self._save_file, CARD)
        btn_save.pack(side=tk.LEFT, padx=(0, 6))

        self._vsep(row2)

        btn_solve = self._mk_btn(row2, "▶  Решить", self._solve, GREEN)
        btn_solve.pack(side=tk.LEFT, padx=(0, 6))
        btn_reset = self._mk_btn(row2, "↺ Сбросить назначение", self._reset_manual, RED)
        btn_reset.pack(side=tk.LEFT)

    def _mk_range_spin(self, parent, var, lo, hi, step, is_chi=False):
        """Spinbox с зажатым диапазоном [lo, hi]. Для χ строго (0,1)."""
        decimals = 2 if is_chi else 2

        def _clamp(e=None):
            try:
                v = float(var.get())
            except (ValueError, tk.TclError):
                v = lo
            var.set(round(max(lo, min(hi, v)), decimals))

        sp = tk.Spinbox(
            parent, from_=lo, to=hi, increment=step, width=5,
            textvariable=var,
            bg=CARD, fg=TEXT, insertbackground=TEXT,
            buttonbackground=CARD, relief=tk.FLAT,
            font=("Consolas", 8),
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT)
        sp.bind("<FocusOut>", _clamp)
        sp.bind("<Return>",   _clamp)
        return sp

    def _vsep(self, parent):
        """Вертикальный разделитель для горизонтальных панелей."""
        tk.Frame(parent, bg=BORDER, width=1
                 ).pack(side=tk.LEFT, fill=tk.Y, pady=6, padx=10)

    def _mk_btn(self, parent, text, cmd, color):
        b = tk.Button(
            parent, text=text, command=cmd,
            bg=color, fg=TEXT,
            activebackground=_hex_lerp(color, "#ffffff", 0.14),
            activeforeground=TEXT,
            relief=tk.FLAT, bd=0, cursor="hand2",
            font=("Segoe UI", 10), padx=12, pady=7)
        return b

    # ── Левая панель ──────────────────────────────────────────────────────────

    def _build_left(self, parent):
        # Вертикальный PanedWindow: матрица A сверху, χ + стратегии снизу
        vpane = tk.PanedWindow(parent, orient=tk.VERTICAL,
                               bg=BORDER, sashwidth=5, sashrelief=tk.FLAT,
                               handlesize=0, opaqueresize=True)
        vpane.pack(fill=tk.BOTH, expand=True)

        # ── Верхняя панель: матрица A ──────────────────────────────────────
        top_pane = tk.Frame(vpane, bg=PANEL)
        vpane.add(top_pane, minsize=80, stretch="always")

        a_lf = self._lf(top_pane, " Матрица C (cᵢⱼ) ")
        a_lf.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
        self._a_sf = _ScrollFrame(a_lf, bg=PANEL)
        self._a_sf.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── Нижняя панель: χ + стратегии ──────────────────────────────────
        bot_pane = tk.Frame(vpane, bg=PANEL)
        vpane.add(bot_pane, minsize=80, stretch="always")

        # Внутренний PanedWindow: χ сверху, эвристики снизу — оба изменяемы
        bot_vpane = tk.PanedWindow(bot_pane, orient=tk.VERTICAL,
                                   bg=BORDER, sashwidth=5, sashrelief=tk.FLAT,
                                   handlesize=0, opaqueresize=True)
        bot_vpane.pack(fill=tk.BOTH, expand=True)

        chi_pane   = tk.Frame(bot_vpane, bg=PANEL)
        strat_pane = tk.Frame(bot_vpane, bg=PANEL)
        bot_vpane.add(chi_pane,   minsize=56,  stretch="never")
        bot_vpane.add(strat_pane, minsize=80,  stretch="always")

        # Вектор χ
        chi_lf = self._lf(chi_pane, " Коэф. уязвимости χᵢ  (0 < χ < 1) ")
        chi_lf.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
        self._chi_sf = _ScrollFrame(chi_lf, bg=PANEL)
        self._chi_sf.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Стратегии — прокручиваемый список
        strat_lf = self._lf(strat_pane, " Эвристики для сравнения ")
        strat_lf.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        btn_row = tk.Frame(strat_lf, bg=PANEL)
        btn_row.pack(fill=tk.X, padx=6, pady=(2, 0))
        btn_sel = tk.Button(btn_row, text="Выбрать все",
                            bg=CARD, fg=TEXT, activebackground=ACCENT,
                            activeforeground=BG, relief=tk.FLAT, padx=8, pady=2,
                            font=("Segoe UI", 8),
                            command=self._select_all)
        btn_sel.pack(side=tk.LEFT, padx=(0, 4))
        btn_desel = tk.Button(btn_row, text="Отменить все",
                              bg=CARD, fg=TEXT, activebackground=ACCENT,
                              activeforeground=BG, relief=tk.FLAT, padx=8, pady=2,
                              font=("Segoe UI", 8),
                              command=self._deselect_all)
        btn_desel.pack(side=tk.LEFT)

        strat_sf = _ScrollFrame(strat_lf, scroll_x=False, scroll_y=True, bg=PANEL)
        strat_sf.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        strat_inner = strat_sf.inner

        for key, label in STRAT_LABELS.items():
            row_f = tk.Frame(strat_inner, bg=PANEL)
            row_f.pack(fill=tk.X, padx=6, pady=2)

            dot_c = tk.Canvas(row_f, width=14, height=14,
                              bg=PANEL, highlightthickness=0)
            dot_c.create_oval(2, 2, 12, 12, fill=STRAT_COLORS[key], outline="")
            dot_c.pack(side=tk.LEFT, padx=(0, 6))

            tk.Checkbutton(
                row_f, text=label, variable=self._strat_vars[key],
                bg=PANEL, fg=TEXT, selectcolor=CARD,
                activebackground=PANEL, activeforeground=TEXT,
                font=("Segoe UI", 10),
                command=self._on_strat_toggle
            ).pack(side=tk.LEFT)

        # Настройка ансамблей — общий параметр m для обоих ансамблей
        ens_f = tk.Frame(strat_inner, bg=CARD, bd=0)
        ens_f.pack(fill=tk.X, padx=6, pady=(4, 6))

        tk.Label(ens_f, text="└ Моделей в ансамбле:",
                 bg=CARD, fg=TEXT_DIM, font=("Segoe UI", 8)
                 ).pack(side=tk.LEFT, padx=(8, 4))

        m_spin = tk.Spinbox(
            ens_f, from_=1, to=500, width=5,
            textvariable=self.ensemble_m,
            bg=CARD, fg=TEXT, insertbackground=TEXT,
            buttonbackground=CARD, relief=tk.FLAT,
            font=("Consolas", 10, "bold"),
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT)
        m_spin.pack(side=tk.LEFT, padx=(0, 8))


        # ── Строка: фиксированный seed ────────────────────────────────────────
        seed_f = tk.Frame(strat_inner, bg=CARD, bd=0)
        seed_f.pack(fill=tk.X, padx=6, pady=(0, 8))

        self._seed_cb = tk.Checkbutton(
            seed_f, text="└ Фикс. seed:", variable=self.ensemble_fixed_seed,
            bg=CARD, fg=TEXT_DIM, selectcolor=GREEN,
            activebackground=CARD, activeforeground=TEXT,
            font=("Segoe UI", 8),
            command=self._on_seed_toggle)
        self._seed_cb.pack(side=tk.LEFT, padx=(8, 4))

        self._seed_spin = tk.Spinbox(
            seed_f, from_=1, to=99999, width=6,
            textvariable=self.ensemble_seed,
            bg=CARD, fg=TEXT_DIM, insertbackground=TEXT,
            buttonbackground=CARD, relief=tk.FLAT,
            font=("Consolas", 10, "bold"),
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT,
            state=tk.DISABLED)
        self._seed_spin.pack(side=tk.LEFT, padx=(0, 6))


    def _lf(self, parent, title):
        return tk.LabelFrame(parent, text=title,
                             bg=PANEL, fg=ACCENT,
                             font=("Segoe UI", 9, "bold"),
                             bd=1, relief=tk.GROOVE, labelanchor="nw")

    # ── Правая панель ─────────────────────────────────────────────────────────

    def _build_right(self, parent):
        # Вертикальный PanedWindow: сетка назначений сверху, результаты снизу
        vpane = tk.PanedWindow(parent, orient=tk.VERTICAL,
                               bg=BORDER, sashwidth=5, sashrelief=tk.FLAT,
                               handlesize=0, opaqueresize=True)
        vpane.pack(fill=tk.BOTH, expand=True)

        # ── Верхняя панель: сетка назначений ──────────────────────────────
        top = tk.Frame(vpane, bg=BG)
        vpane.add(top, minsize=150, stretch="always")

        top.grid_rowconfigure(1, weight=1)
        top.grid_columnconfigure(0, weight=1)

        hdr = tk.Frame(top, bg=BG)
        hdr.grid(row=0, column=0, sticky="ew", pady=(2, 4))

        tk.Label(hdr, text="Сетка назначений",
                 bg=BG, fg=TEXT, font=("Segoe UI", 12, "bold")
                 ).pack(side=tk.LEFT)
        tk.Label(hdr,
                 text="  кликните на ячейку для ручного выбора; повторный клик — снять",
                 bg=BG, fg=TEXT_DIM, font=("Segoe UI", 9)
                 ).pack(side=tk.LEFT)

        # Переключатель A / B — справа в заголовке, всегда на виду
        tog_f = tk.Frame(hdr, bg=BG)
        tog_f.pack(side=tk.RIGHT, padx=(0, 4))
        tk.Label(tog_f, text="Показать:",
                 bg=BG, fg=TEXT_DIM, font=("Segoe UI", 9)
                 ).pack(side=tk.LEFT, padx=(0, 4))
        rb_g = tk.Radiobutton(tog_f, text="матрицу G", variable=self.show_var, value="B",
                               bg=BG, fg=TEXT, activebackground=BG, activeforeground=ACCENT,
                               font=("Segoe UI", 9), indicatoron=False, selectcolor=ACCENT,
                               padx=8, pady=3, relief=tk.FLAT, bd=1, command=self._redraw)
        rb_g.pack(side=tk.LEFT, padx=2)
        rb_d = tk.Radiobutton(tog_f, text="матрицу D", variable=self.show_var, value="D",
                               bg=BG, fg=TEXT, activebackground=BG, activeforeground=ACCENT,
                               font=("Segoe UI", 9), indicatoron=False, selectcolor=ACCENT,
                               padx=8, pady=3, relief=tk.FLAT, bd=1, command=self._redraw)
        rb_d.pack(side=tk.LEFT, padx=2)
        rb_c = tk.Radiobutton(tog_f, text="матрицу C", variable=self.show_var, value="A",
                               bg=BG, fg=TEXT, activebackground=BG, activeforeground=ACCENT,
                               font=("Segoe UI", 9), indicatoron=False, selectcolor=ACCENT,
                               padx=8, pady=3, relief=tk.FLAT, bd=1, command=self._redraw)
        rb_c.pack(side=tk.LEFT, padx=2)

        wrap = tk.Frame(top, bg=PANEL, bd=1, relief=tk.GROOVE)
        wrap.grid(row=1, column=0, sticky="nsew")
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)

        self._canv = tk.Canvas(wrap, bg=CARD, highlightthickness=0, cursor="hand2")
        sb_v = tk.Scrollbar(wrap, orient=tk.VERTICAL, command=self._canv.yview)
        sb_h = tk.Scrollbar(wrap, orient=tk.HORIZONTAL, command=self._canv.xview)
        self._canv.configure(yscrollcommand=sb_v.set, xscrollcommand=sb_h.set)
        sb_v.grid(row=0, column=1, sticky="ns")
        sb_h.grid(row=1, column=0, sticky="ew")
        self._canv.grid(row=0, column=0, sticky="nsew")

        self._canv.bind("<Button-1>", self._on_click)
        self._canv.bind("<Motion>",   self._on_motion)
        self._canv.bind("<Leave>",    self._on_leave_event)

        # ── Нижняя панель: результаты ──────────────────────────────────────
        res_outer = tk.Frame(vpane, bg=PANEL, bd=1, relief=tk.GROOVE)
        vpane.add(res_outer, minsize=80, stretch="always")
        self._build_results(res_outer)

    def _build_results(self, parent):
        tk.Label(parent, text="  Результаты сравнения",
                 bg=PANEL, fg=ACCENT, font=("Segoe UI", 10, "bold"),
                 anchor="w").pack(fill=tk.X, pady=(6, 2))
        self._res_sf = _ScrollFrame(parent, scroll_x=False, scroll_y=True, bg=PANEL)
        self._res_sf.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))
        # _res_frame указывает на inner — весь существующий код добавляет виджеты туда
        self._res_frame = self._res_sf.inner

    # ── Управление входными данными ───────────────────────────────────────────

    def _rebuild_inputs(self, n):
        for w in self._a_sf.inner.winfo_children():
            w.destroy()
        for w in self._chi_sf.inner.winfo_children():
            w.destroy()
        self._a_entries = []
        self._chi_entries = []

        e_kw = dict(
            bg=CARD, fg=TEXT, insertbackground=TEXT,
            relief=tk.FLAT, font=("Consolas", 10), justify="center",
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT, width=7)

        # Заголовки столбцов матрицы A
        tk.Label(self._a_sf.inner, text="", bg=PANEL, width=3
                 ).grid(row=0, column=0)
        for j in range(n):
            tk.Label(self._a_sf.inner, text=f"t{j+1}",
                     bg=PANEL, fg=TEXT_DIM, font=("Segoe UI", 8),
                     width=7, anchor="center"
                     ).grid(row=0, column=j + 1, padx=1)

        # Строки матрицы A
        for i in range(n):
            tk.Label(self._a_sf.inner, text=f"w{i+1}",
                     bg=PANEL, fg=TEXT_DIM, font=("Segoe UI", 8),
                     width=3, anchor="e"
                     ).grid(row=i + 1, column=0, padx=(4, 2), pady=1)
            row_ents = []
            for j in range(n):
                e = tk.Entry(self._a_sf.inner, **e_kw)
                e.grid(row=i + 1, column=j + 1, padx=1, pady=1)
                e.insert(0, "0.00")
                e.bind("<FocusOut>", lambda _ev: self._on_entry_change())
                row_ents.append(e)
            self._a_entries.append(row_ents)

        # χ-вектор
        tk.Label(self._chi_sf.inner, text="χ:",
                 bg=PANEL, fg=TEXT_DIM, font=("Segoe UI", 9), width=2
                 ).grid(row=0, column=0, padx=(4, 4), pady=4)
        for i in range(n):
            tk.Label(self._chi_sf.inner, text=f"w{i+1}",
                     bg=PANEL, fg=TEXT_DIM, font=("Segoe UI", 7),
                     anchor="center"
                     ).grid(row=0, column=2 * i + 1, padx=(2, 0))
            e = tk.Entry(self._chi_sf.inner,
                         bg=CARD, fg=TEXT, insertbackground=TEXT,
                         relief=tk.FLAT, font=("Consolas", 10), justify="center",
                         highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=ACCENT, width=6)
            e.grid(row=0, column=2 * i + 2, padx=1, pady=4)
            e.insert(0, "0.50")
            e.bind("<FocusOut>", lambda _ev: self._on_entry_change())
            self._chi_entries.append(e)

    def _fill_entries(self, C, chi):
        n = C.shape[0]
        for i in range(n):
            for j in range(n):
                self._a_entries[i][j].delete(0, tk.END)
                self._a_entries[i][j].insert(0, f"{C[i, j]:.2f}")
        for i, e in enumerate(self._chi_entries):
            e.delete(0, tk.END)
            e.insert(0, f"{chi[i]:.3f}")

    def _read_inputs(self):
        n = self.n_var.get()
        try:
            tmp_C = []
            for i in range(n):
                row = []
                for j in range(n):
                    row.append(float(self._a_entries[i][j].get()))
                tmp_C.append(row)
            C = np.array(tmp_C)
            tmp_chi = []
            for e in self._chi_entries:
                tmp_chi.append(float(e.get()))
            chi = np.array(tmp_chi)
        except ValueError as ex:
            messagebox.showerror("Ошибка ввода", f"Неверный формат:\n{ex}")
            return None, None
        if np.any(chi <= 0) or np.any(chi >= 1):
            messagebox.showerror("Ошибка", "Все χᵢ должны быть в диапазоне (0, 1)")
            return None, None
        return C, chi

    def _on_entry_change(self):
        C, chi = self._read_inputs()
        if C is not None:
            self._C, self._chi = C, chi
            self._G = build_G(C, chi)
            self._D = build_D(C, chi)
            self._strat_res.clear()
            self._redraw()
            self._update_results()

    # ── Действия ─────────────────────────────────────────────────────────────

    def _on_n_change_event(self, event):
        self._on_n_change()

    def _on_n_change(self):
        n = max(2, min(20, self.n_var.get()))
        self.n_var.set(n)
        self._C = self._chi = self._G = self._D = None
        self._manual.clear()
        self._strat_res.clear()
        self._rebuild_inputs(n)
        self._canv.delete("all")
        self._update_results()

    def _do_random(self):
        n = self.n_var.get()

        # Диапазоны с защитой от некорректных значений
        a_lo  = max(0.01, self.a_low.get())
        a_hi  = max(a_lo + 0.01, self.a_high.get())
        c_lo  = max(0.01, min(0.98, self.chi_low.get()))
        c_hi  = max(c_lo + 0.01, min(0.99, self.chi_high.get()))

        C = np.random.uniform(a_lo, a_hi, (n, n))
        mode = self.gen_mode.get()
        if mode == "desc":
            for i in range(n):
                C[i] = np.sort(C[i])[::-1]
        elif mode == "asc":
            for i in range(n):
                C[i] = np.sort(C[i])
        chi = np.random.uniform(c_lo, c_hi, n)

        self._rebuild_inputs(n)
        self._fill_entries(C, chi)
        self._C, self._chi = C, chi
        self._G = build_G(C, chi)
        self._D = build_D(C, chi)
        self._manual.clear()
        self._strat_res.clear()
        self._redraw()
        self._update_results()

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Открыть задачу",
            filetypes=[("JSON", "*.json"), ("CSV", "*.csv"), ("Все", "*.*")])
        if not path:
            return
        try:
            if path.lower().endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                key = "C" if "C" in data else "A"
                C   = np.array(data[key],  dtype=float)
                chi = np.array(data["chi"], dtype=float)
            else:
                import csv as _csv
                with open(path, "r", encoding="utf-8") as f:
                    rows = list(_csv.reader(f))
                tmp_chi = []
                for x in rows[0]:
                    tmp_chi.append(float(x))
                chi = np.array(tmp_chi, dtype=float)
                tmp_C = []
                for row in rows[1:]:
                    tmp_row = []
                    for x in row:
                        tmp_row.append(float(x))
                    tmp_C.append(tmp_row)
                C = np.array(tmp_C, dtype=float)
            n = C.shape[0]
            self.n_var.set(n)
            self._rebuild_inputs(n)
            self._fill_entries(C, chi)
            self._C, self._chi = C, chi
            self._G = build_G(C, chi)
            self._D = build_D(C, chi)
            self._manual.clear()
            self._strat_res.clear()
            self._redraw()
            self._update_results()
        except Exception as ex:
            messagebox.showerror("Ошибка загрузки", str(ex))

    def _save_file(self):
        C, chi = self._read_inputs()
        if C is None:
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить задачу",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("CSV", "*.csv")])
        if not path:
            return
        try:
            if path.lower().endswith(".json"):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"A": C.tolist(), "chi": chi.tolist()},
                              f, indent=2, ensure_ascii=False)
            else:
                import csv as _csv
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = _csv.writer(f)
                    w.writerow(chi.tolist())
                    for row in C.tolist():
                        w.writerow(row)
            messagebox.showinfo("Сохранено", f"Файл:\n{path}")
        except Exception as ex:
            messagebox.showerror("Ошибка сохранения", str(ex))

    def _reset_manual(self):
        self._manual.clear()
        self._redraw()
        self._update_results()

    def _solve(self):
        C, chi = self._read_inputs()
        if C is None:
            return
        self._C, self._chi = C, chi
        self._G = build_G(C, chi)
        self._D = build_D(C, chi)
        self._strat_res.clear()

        m    = max(1, self.ensemble_m.get())
        seed = self.ensemble_seed.get() if self.ensemble_fixed_seed.get() else None
        ensemble_keys = ("ensemble", "ensemble_chi")

        for name, fn in ALL_STRATS.items():
            if self._strat_vars[name].get():
                t0 = time.perf_counter()
                sched = fn(C, chi, m, seed) if name in ensemble_keys else fn(C, chi)
                t1 = time.perf_counter()
                self._strat_res[name] = {
                    "sched":  sched,
                    "profit": calc_profit(C, chi, sched),
                    "time":   t1 - t0,
                    "m":      m if name in ensemble_keys else None,
                    "seed":   seed if name in ensemble_keys else None,
                }

        self._redraw()
        self._update_results()

    def _select_all(self):
        for v in self._strat_vars.values():
            v.set(True)

    def _deselect_all(self):
        for v in self._strat_vars.values():
            v.set(False)

    def _on_strat_toggle(self):
        self._redraw()
        self._update_results()


    def _on_seed_toggle(self):
        is_fixed = self.ensemble_fixed_seed.get()
        self._seed_spin.config(
            state=tk.NORMAL if is_fixed else tk.DISABLED,
            fg=TEXT if is_fixed else TEXT_DIM)

    # ── Сетка назначений (Canvas) ─────────────────────────────────────────────

    def _redraw(self):
        _sv = self.show_var.get()
        if   _sv == "B" and self._G is not None: M = self._G
        elif _sv == "D" and self._D is not None: M = self._D
        else:                                     M = self._C
        if M is None:
            return
        n = self.n_var.get()
        if len(self._a_entries) != n:
            return

        show_b = (_sv == "B")
        vmin, vmax = float(M.min()), float(M.max())
        used_w = set(self._manual.values())
        used_s = set(self._manual.keys())

        show_chi_col = (_sv == "A") and (self._chi is not None)

        has_legend = False
        for k in ALL_STRATS:
            if self._strat_vars[k].get() and k in self._strat_res:
                has_legend = True
                break
        legend_h = 22 if has_legend else 0
        cw = HDR_W + n * CELL_W + (CHI_W if show_chi_col else 0) + 4
        ch = HDR_H + n * CELL_H + 4 + legend_h

        self._canv.configure(scrollregion=(0, 0, cw, ch))
        self._canv.delete("all")

        # Угол
        self._canv.create_rectangle(0, 0, HDR_W, HDR_H,
                                    fill=HDR_BG, outline=BORDER)
        if _sv == "B":
            label = "G"
        elif _sv == "D":
            label = "D"
        else:
            label = "C"
        self._canv.create_text(HDR_W // 2, HDR_H // 2, text=label,
                               fill=TEXT_DIM, font=("Segoe UI", 8, "italic"))

        # Заголовки столбцов (слоты t)
        for j in range(n):
            x0 = HDR_W + j * CELL_W
            used = j in used_s
            self._canv.create_rectangle(x0, 0, x0 + CELL_W, HDR_H,
                                        fill=HDR_BG, outline=BORDER)
            self._canv.create_text(
                x0 + CELL_W // 2, HDR_H // 2,
                text=f"t{j+1}",
                fill=ACCENT if used else TEXT_DIM,
                font=("Segoe UI", 9, "bold" if used else "normal"))

        # Заголовок колонки χᵢ
        if show_chi_col:
            cx0 = HDR_W + n * CELL_W
            self._canv.create_rectangle(cx0, 0, cx0 + CHI_W, HDR_H,
                                        fill=HDR_BG, outline=BORDER)
            self._canv.create_text(cx0 + CHI_W // 2, HDR_H // 2,
                                   text="χᵢ", fill=ACCENT,
                                   font=("Segoe UI", 9, "bold"))

        # Заголовки строк (работники w) + chi-ячейки
        for i in range(n):
            y0 = HDR_H + i * CELL_H
            used = i in used_w
            self._canv.create_rectangle(0, y0, HDR_W, y0 + CELL_H,
                                        fill=HDR_BG, outline=BORDER)
            self._canv.create_text(
                HDR_W // 2, y0 + CELL_H // 2,
                text=f"w{i+1}",
                fill=ACCENT if used else TEXT_DIM,
                font=("Segoe UI", 9, "bold" if used else "normal"))

            if show_chi_col:
                self._draw_chi_cell(i, y0)

        # Ячейки матрицы
        for i in range(n):
            for j in range(n):
                self._draw_cell(i, j, M, vmin, vmax, used_w, used_s)

        # Легенда
        if has_legend:
            self._draw_legend(ch - legend_h + 2)

    def _draw_cell(self, i, j, M, vmin, vmax, used_w, used_s):
        x0 = HDR_W + j * CELL_W
        y0 = HDR_H + i * CELL_H
        x1, y1 = x0 + CELL_W, y0 + CELL_H
        cx = x0 + CELL_W // 2
        cy = y0 + CELL_H // 2

        is_sel = (j in self._manual and self._manual[j] == i)
        is_dim = (not is_sel) and (i in used_w or j in used_s)
        is_hov = (self._hover == (i, j))

        # Фон и рамка
        if is_sel:
            bg, bd, bw = SEL_BG, SEL_BD, 2
        elif is_dim:
            bg, bd, bw = DIM_BG, "#1a1f26", 1
        elif is_hov:
            bg = _hex_lerp(_val_color(M[i, j], vmin, vmax), "#ffffff", 0.09)
            bd, bw = ACCENT, 1
        else:
            bg, bd, bw = _val_color(M[i, j], vmin, vmax), BORDER, 1

        # Внутренний прямоугольник (без рамки)
        self._canv.create_rectangle(x0 + bw, y0 + bw, x1 - bw, y1 - bw,
                                    fill=bg, outline="")
        # Рамка
        self._canv.create_rectangle(x0, y0, x1, y1,
                                    fill="", outline=bd, width=bw)

        # Значение (немного сдвинуто вниз — вверху место для точек стратегий)
        val_str = f"{M[i, j]:.1f}"
        txt_col = DIM_TXT if is_dim else (GREEN if is_sel else TEXT)
        txt_y   = cy + 6

        self._canv.create_text(cx, txt_y, text=val_str,
                               fill=txt_col,
                               font=("Consolas", 10, "bold" if is_sel else "normal"))

        # Зачёркивание для затемнённых ячеек
        if is_dim:
            hw = len(val_str) * 3
            self._canv.create_line(cx - hw, txt_y, cx + hw, txt_y,
                                   fill=DIM_TXT, width=1)

        # Галочка для выбранной ячейки
        if is_sel:
            self._canv.create_text(cx, y0 + 9,
                                   text="✓", fill=GREEN,
                                   font=("Segoe UI", 9, "bold"))

        # Точки стратегий
        self._draw_dots(x0, y0, i, j)

    def _draw_chi_cell(self, i, y0):
        """Ячейка колонки χᵢ справа от матрицы A."""
        chi_val = float(self._chi[i])
        x0 = HDR_W + self.n_var.get() * CELL_W
        x1, y1 = x0 + CHI_W, y0 + CELL_H

        # Цвет: низкий χ (уязвимее) → красноватый, высокий → зеленоватый
        t = chi_val          # уже в (0,1)
        bg = _hex_lerp("#5c1a1a", "#1a5c2a", t)

        self._canv.create_rectangle(x0, y0, x1, y1, fill=bg, outline=BORDER)
        self._canv.create_text(
            x0 + CHI_W // 2, y0 + CELL_H // 2,
            text=f"{chi_val:.2f}",
            fill=TEXT, font=("Consolas", 10, "bold"))

    def _draw_dots(self, x0, y0, i, j):
        """Цветные точки для каждой стратегии, назначившей worker i → slot j."""
        active = []
        for k, v in self._strat_vars.items():
            if v.get():
                active.append(k)
        if not active:
            return
        total_w = len(active) * (DOT_D + DOT_G) - DOT_G
        start_x = x0 + max(2, (CELL_W - total_w) // 2)
        for k, name in enumerate(active):
            if name not in self._strat_res:
                continue
            sched = self._strat_res[name]["sched"]
            if j < len(sched) and sched[j] == i:
                dx = start_x + k * (DOT_D + DOT_G)
                dy = y0 + 3
                self._canv.create_oval(dx, dy, dx + DOT_D, dy + DOT_D,
                                       fill=STRAT_COLORS[name], outline="")

    def _draw_legend(self, y):
        """Легенда цветов стратегий под сеткой."""
        x = HDR_W + 4
        self._canv.create_text(x, y + 8, text="▶",
                               fill=TEXT_DIM, font=("Segoe UI", 7), anchor="w")
        x += 14
        for name in ALL_STRATS:
            if not self._strat_vars[name].get() or name not in self._strat_res:
                continue
            self._canv.create_oval(x, y + 3, x + DOT_D, y + 3 + DOT_D,
                                   fill=STRAT_COLORS[name], outline="")
            self._canv.create_text(x + DOT_D + 3, y + 8,
                                   text=STRAT_LABELS[name],
                                   fill=TEXT_DIM, font=("Segoe UI", 7), anchor="w")
            x += DOT_D + 3 + len(STRAT_LABELS[name]) * 5 + 8

    # ── Обработка мыши на Canvas ──────────────────────────────────────────────

    def _on_click(self, event):
        if self._G is None and self._C is None:
            return
        n = self.n_var.get()
        x = self._canv.canvasx(event.x)
        y = self._canv.canvasy(event.y)
        j = int((x - HDR_W) // CELL_W)
        i = int((y - HDR_H) // CELL_H)
        if not (0 <= i < n and 0 <= j < n):
            return

        if j in self._manual and self._manual[j] == i:
            del self._manual[j]
        else:
            old_j = None
            for jj, ii in self._manual.items():
                if ii == i:
                    old_j = jj
                    break
            if old_j is not None:
                del self._manual[old_j]
            self._manual[j] = i

        self._redraw()
        self._update_results()

    def _on_motion(self, event):
        if self._C is None and self._G is None:
            return
        n = self.n_var.get()
        x = self._canv.canvasx(event.x)
        y = self._canv.canvasy(event.y)
        j = int((x - HDR_W) // CELL_W)
        i = int((y - HDR_H) // CELL_H)
        new_hov = (i, j) if (0 <= i < n and 0 <= j < n) else None
        if new_hov != self._hover:
            self._hover = new_hov
            self._redraw()

    def _on_leave_event(self, event):
        self._on_leave()

    def _on_leave(self):
        if self._hover is not None:
            self._hover = None
            self._redraw()

    # ── Панель результатов ────────────────────────────────────────────────────

    def _update_results(self):
        for w in self._res_frame.winfo_children():
            w.destroy()

        if self._C is None or self._G is None:
            return

        n = self.n_var.get()
        rows = []

        # Ручное назначение
        if self._manual:
            filled = len(self._manual)
            if filled == n:
                sched = [self._manual[j] for j in range(n)]
                p = calc_profit(self._C, self._chi, sched)
                rows.append(("👤 Ваше", sched, p, "#aaaaaa"))
            else:
                rows.append((f"👤 Ваше ({filled}/{n})", [], None, "#666666"))

        for name in ALL_STRATS:
            if not self._strat_vars[name].get():
                continue
            if name not in self._strat_res:
                continue
            d = self._strat_res[name]
            tmp_label = STRAT_LABELS[name]
            if d.get("m") is not None:
                tmp_label = tmp_label + f" (m={d['m']})"
            tmp_sched  = d["sched"]
            tmp_profit = d["profit"]
            tmp_color  = STRAT_COLORS[name]
            rows.append((tmp_label, tmp_sched, tmp_profit, tmp_color))

        if not rows:
            tk.Label(self._res_frame,
                     text="Нажмите  ▶ Решить  для запуска эвристик",
                     bg=PANEL, fg=TEXT_DIM, font=("Segoe UI", 9)
                     ).pack(padx=10, pady=6)
            return

        if "hungarian" in self._strat_res:
            ref_profit = self._strat_res["hungarian"]["profit"]
        else:
            ref_profit = None

        best_profit = None
        for r in rows:
            if r[2] is not None:
                if best_profit is None or r[2] > best_profit:
                    best_profit = r[2]

        def sort_key(r):
            if r[2] is None:
                return -1e18
            return r[2]
        rows.sort(key=sort_key, reverse=True)

        # ── Единая таблица: один grid-контейнер для выравнивания колонок ──
        tbl = tk.Frame(self._res_frame, bg=PANEL)
        tbl.pack(fill=tk.BOTH, expand=True)

        # col 0: цветная полоска (фикс)
        # col 1: название         (фикс ~170px)
        # col 2: прибыль          (фикс ~100px, выравнивание вправо)
        # col 3: Δ%               (фикс ~72px,  выравнивание вправо)
        # (расписание занимает отдельную строку — cols 1..3)
        tbl.grid_columnconfigure(0, minsize=10)
        tbl.grid_columnconfigure(1, minsize=172, weight=0)
        tbl.grid_columnconfigure(2, minsize=106, weight=0)
        tbl.grid_columnconfigure(3, minsize=74,  weight=0)

        # Заголовок (строка 0)
        for col, (txt, anch) in enumerate([
            ("",           "center"),
            ("Метод",      "w"),
            ("Прибыль",    "e"),
            ("Δ% vs Венг.","e"),
        ]):
            tk.Label(tbl, text=txt,
                     bg=CARD, fg=TEXT_DIM, font=("Segoe UI", 8, "bold"),
                     anchor=anch, padx=6, pady=4
                     ).grid(row=0, column=col, sticky="ew", padx=1, pady=(0, 3))

        for idx, (name, sched, profit, color) in enumerate(rows):
            is_best = (profit is not None and profit == best_profit)
            self._result_row(tbl, idx, name, sched, profit,
                             ref_profit, color, is_best, n)

    def _result_row(self, tbl, idx, name, sched, profit,
                    ref_profit, color, is_best, n):
        """Два ряда grid на одну запись: [полоска | имя | прибыль | Δ%]
                                          [       | расписание → → →   ]"""
        r0 = 1 + idx * 2      # первая строка записи
        r1 = r0 + 1           # вторая строка — расписание

        row_bg = _hex_lerp(PANEL, "#1e2836", 0.6) if is_best else PANEL

        # Цветная полоска (spans 2 rows)
        bar_c = tk.Canvas(tbl, width=6, height=38,
                          bg=row_bg, highlightthickness=0)
        bar_c.create_rectangle(0, 3, 6, 35, fill=color, outline="")
        bar_c.grid(row=r0, column=0, rowspan=2, padx=(4, 3), sticky="ns")

        # Название
        tk.Label(tbl, text=name,
                 bg=row_bg,
                 fg=GREEN if is_best else TEXT,
                 font=("Segoe UI", 9, "bold") if is_best else ("Segoe UI", 9),
                 anchor="w", padx=4
                 ).grid(row=r0, column=1, sticky="ew", pady=(3, 0))

        # Прибыль
        if profit is not None:
            tk.Label(tbl, text=f"{profit:,.2f}",
                     bg=row_bg,
                     fg=GREEN if is_best else TEXT,
                     font=("Consolas", 11, "bold") if is_best else ("Consolas", 10),
                     anchor="e", padx=6
                     ).grid(row=r0, column=2, sticky="ew", pady=(3, 0))
        else:
            tk.Label(tbl, text="—",
                     bg=row_bg, fg=TEXT_DIM,
                     font=("Consolas", 10), anchor="e", padx=6
                     ).grid(row=r0, column=2, sticky="ew", pady=(3, 0))

        # Δ%
        if profit is not None and ref_profit is not None and abs(ref_profit) > 1e-9:
            delta = (profit - ref_profit) / abs(ref_profit) * 100
            tk.Label(tbl, text=f"{delta:+.2f}%",
                     bg=row_bg,
                     fg=GREEN if delta >= -0.001 else RED,
                     font=("Consolas", 10, "bold") if is_best else ("Consolas", 9),
                     anchor="e", padx=6
                     ).grid(row=r0, column=3, sticky="ew", pady=(3, 0))

        # Расписание — вторая строка, spans 3 колонки
        if sched:
            parts = []
            for j in range(len(sched)):
                if sched[j] >= 0:
                    parts.append(f"t{j+1}→w{sched[j]+1}")
            sched_str = "   ".join(parts)
        else:
            sched_str = "  нет данных"

        tk.Label(tbl, text=sched_str,
                 bg=row_bg, fg="#9eacbd",
                 font=("Consolas", 9),
                 anchor="w", padx=6
                 ).grid(row=r1, column=1, columnspan=3,
                        sticky="ew", pady=(0, 4))

        # Тонкий разделитель после каждой записи
        sep = tk.Frame(tbl, bg=BORDER, height=1)
        sep.grid(row=r1 + 1, column=0, columnspan=4,
                 sticky="ew", padx=4, pady=0)


# ── Точка входа ──────────────────────────────────────────────────────────────

def main():
    app = AssignmentApp()
    app.mainloop()


if __name__ == "__main__":
    main()
