#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_mpms.py -- plot and fit Quantum Design MPMS3 (.dat) magnetometry data.

Point it at a folder, pick the files you want, and it produces publication-style
figures (and optionally re-plottable .txt files and Curie-Weiss fit results).

    python plot_mpms.py "C:\\path\\to\\MPMS"
    python plot_mpms.py . --files "NaRhO2_MattW*MvH*" --formula NaRhO2 --ion Rh
    python plot_mpms.py . --all --fit --fit-range 150 300 --export-txt

Expected file naming (tokens may appear in any order after the sample id):
    <sample-id>_<mass>mg_MvT_<field>Oe_<FC|ZFC>.dat
    <sample-id>_<mass>mg_MvH_<temperature>K.dat
e.g. NaRhO2_MattW_23p35mg_MvT_500Oe_ZFC.dat
     NaRhO2_MattW_23p35mg_MvH_2K.dat
The sample id is everything before the _MvT_ / _MvH_ token; files sharing a
sample id are grouped onto the same figures.

Requires: numpy, pandas, matplotlib, scipy  (Python 3.9+)
"""

from __future__ import annotations

__version__ = "1.1"

import argparse
import fnmatch
import io
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from mpl_toolkits.axes_grid1 import Divider, Size
from scipy.optimize import curve_fit

# --------------------------------------------------------------------------- #
# Physical constants (CODATA 2018, cgs)
# --------------------------------------------------------------------------- #
N_A = 6.02214076e23          # 1/mol
MU_B = 9.2740100783e-21      # erg/G  (= emu)
EMU_PER_MOL_PER_MUB = N_A * MU_B   # 5584.94 emu/mol  ==  1 muB per formula unit

CONFIG_NAME = ".mpms_plot_config.json"

# --------------------------------------------------------------------------- #
# Standard atomic weights (IUPAC 2021 conventional values)
# --------------------------------------------------------------------------- #
ATOMIC_WEIGHTS = {
    "H": 1.008, "He": 4.002602, "Li": 6.94, "Be": 9.0121831, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998403163, "Ne": 20.1797,
    "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815385, "Si": 28.085,
    "P": 30.973761998, "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.0983,
    "Ca": 40.078, "Sc": 44.955908, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961,
    "Mn": 54.938044, "Fe": 55.845, "Co": 58.933194, "Ni": 58.6934,
    "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.921595,
    "Se": 78.971, "Br": 79.904, "Kr": 83.798, "Rb": 85.4678, "Sr": 87.62,
    "Y": 88.90584, "Zr": 91.224, "Nb": 92.90637, "Mo": 95.95, "Tc": 98.0,
    "Ru": 101.07, "Rh": 102.90550, "Pd": 106.42, "Ag": 107.8682,
    "Cd": 112.414, "In": 114.818, "Sn": 118.710, "Sb": 121.760, "Te": 127.60,
    "I": 126.90447, "Xe": 131.293, "Cs": 132.90545196, "Ba": 137.327,
    "La": 138.90547, "Ce": 140.116, "Pr": 140.90766, "Nd": 144.242,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.92535,
    "Dy": 162.500, "Ho": 164.93033, "Er": 167.259, "Tm": 168.93422,
    "Yb": 173.045, "Lu": 174.9668, "Hf": 178.49, "Ta": 180.94788, "W": 183.84,
    "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084,
    "Au": 196.966569, "Hg": 200.592, "Tl": 204.38, "Pb": 207.2,
    "Bi": 208.98040, "Po": 209.0, "At": 210.0, "Rn": 222.0, "Fr": 223.0,
    "Ra": 226.0, "Ac": 227.0, "Th": 232.0377, "Pa": 231.03588, "U": 238.02891,
    "Np": 237.0, "Pu": 244.0,
}

# Approximate Pascal's constants, per ATOM, in 1e-6 emu/mol, for the most common
# oxidation state of each element in an oxide.  These are estimates only -- if
# the diamagnetic correction matters for your result, use --chi-dia to supply
# your own value in emu/mol.
PASCAL_1E6 = {
    "H": -2.93, "Li": -1.0, "Be": -0.4, "B": -3.0, "C": -6.0, "N": -5.6,
    "O": -12.0, "F": -9.1, "Na": -6.8, "Mg": -5.0, "Al": -2.0, "Si": -1.0,
    "P": -4.0, "S": -30.0, "Cl": -23.4, "K": -14.9, "Ca": -10.4, "Sc": -6.0,
    "Ti": -5.0, "V": -7.0, "Cr": -11.0, "Mn": -14.0, "Fe": -12.0,
    "Co": -12.0, "Ni": -12.0, "Cu": -12.8, "Zn": -15.0, "Ga": -8.0,
    "Ge": -7.0, "As": -20.9, "Se": -23.0, "Br": -34.6, "Rb": -22.5,
    "Sr": -19.0, "Y": -12.0, "Zr": -10.0, "Nb": -9.0, "Mo": -12.0,
    "Ru": -23.0, "Rh": -22.0, "Pd": -25.0, "Ag": -28.0, "Cd": -24.0,
    "In": -19.0, "Sn": -20.0, "Sb": -74.0, "Te": -37.0, "I": -50.6,
    "Cs": -35.0, "Ba": -26.5, "La": -20.0, "Ce": -20.0, "Pr": -20.0,
    "Nd": -20.0, "Sm": -20.0, "Eu": -22.0, "Gd": -20.0, "Tb": -19.0,
    "Dy": -19.0, "Ho": -19.0, "Er": -18.0, "Tm": -18.0, "Yb": -20.0,
    "Lu": -17.0, "Hf": -16.0, "Ta": -14.0, "W": -13.0, "Re": -12.0,
    "Os": -36.0, "Ir": -50.0, "Pt": -40.0, "Au": -45.0, "Hg": -40.0,
    "Tl": -35.0, "Pb": -32.0, "Bi": -26.0, "Th": -23.0, "U": -35.0,
}

FORMULA_HELP = """\
Formula syntax
--------------
  * Element symbols are case sensitive:  Rh  not  RH  or  rh
  * Integer or decimal subscripts:       NaRhO2 , Na0.5RhO2 , Li1.33Mn0.67O2
  * Parentheses, nested, with subscripts: (Na0.5K0.5)RhO2 , Ba(FeO2)2
  * Whitespace is ignored:               Na 0.5 Rh O2
  * Write the resolved composition -- "Na1-xRhO2" cannot be parsed, use e.g.
    Na0.5RhO2 for x = 0.5.
"""

# --------------------------------------------------------------------------- #
# Formula parsing
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"([A-Z][a-z]?)\s*(\d*\.?\d*)|(\()|(\)\s*(\d*\.?\d*))")


def parse_formula(formula: str) -> "OrderedDict[str, float]":
    """Parse a chemical formula into an ordered {element: count} mapping."""
    s = re.sub(r"\s+", "", formula)
    if not s:
        raise ValueError("empty formula")

    stack = [OrderedDict()]
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "(":
            stack.append(OrderedDict())
            i += 1
        elif ch == ")":
            i += 1
            m = re.match(r"\d*\.?\d*", s[i:])
            mult = float(m.group()) if m.group() else 1.0
            i += m.end()
            grp = stack.pop()
            if not stack:
                raise ValueError("unbalanced ')' in formula")
            for el, n in grp.items():
                stack[-1][el] = stack[-1].get(el, 0.0) + n * mult
        else:
            m = re.match(r"([A-Z][a-z]?)(\d*\.?\d*)", s[i:])
            if not m:
                raise ValueError(f"cannot parse formula at '{s[i:]}'")
            el, num = m.group(1), m.group(2)
            if el not in ATOMIC_WEIGHTS:
                raise ValueError(f"unknown element '{el}' (symbols are case sensitive)")
            n = float(num) if num else 1.0
            stack[-1][el] = stack[-1].get(el, 0.0) + n
            i += m.end()
    if len(stack) != 1:
        raise ValueError("unbalanced '(' in formula")
    return stack[0]


def molar_mass(counts) -> float:
    return sum(ATOMIC_WEIGHTS[el] * n for el, n in counts.items())


def pascal_chi_dia(counts) -> float:
    """Rough Pascal's-constant diamagnetic susceptibility, emu/mol f.u."""
    total = 0.0
    missing = []
    for el, n in counts.items():
        if el in PASCAL_1E6:
            total += PASCAL_1E6[el] * n
        else:
            missing.append(el)
    if missing:
        print(f"  ! no Pascal constant tabulated for {', '.join(missing)} "
              f"-- excluded from the estimate")
    return total * 1e-6


def format_formula(counts) -> str:
    out = []
    for el, n in counts.items():
        out.append(el if abs(n - 1.0) < 1e-12 else f"{el}{n:g}")
    return "".join(out)


# --------------------------------------------------------------------------- #
# .dat reading
# --------------------------------------------------------------------------- #
MOMENT_PREFERENCE = [
    "Moment (emu)",
    "DC Moment Free Ctr (emu)",
    "DC Moment Fixed Ctr (emu)",
]
ERR_FOR_MOMENT = {
    "Moment (emu)": "M. Std. Err. (emu)",
    "DC Moment Free Ctr (emu)": "DC Moment Err Free Ctr (emu)",
    "DC Moment Fixed Ctr (emu)": "DC Moment Err Fixed Ctr (emu)",
}


def read_dat(path: Path):
    """Return (header_info dict, DataFrame) for an MPMS3 .dat file."""
    text = path.read_text(encoding="latin-1")
    if "[Data]" not in text:
        raise ValueError(f"{path.name}: no [Data] section -- is this an MPMS .dat file?")
    head, body = text.split("[Data]", 1)

    info = {}
    for line in head.splitlines():
        parts = line.split(",")
        if parts and parts[0].strip().upper() == "INFO" and len(parts) >= 3:
            key = parts[-1].strip()
            val = ",".join(parts[1:-1]).strip()
            if key:
                info[key] = val
        elif parts and parts[0].strip().upper() == "FILEOPENTIME" and len(parts) >= 3:
            info["FILEOPENTIME"] = ",".join(parts[1:]).strip()

    df = pd.read_csv(io.StringIO(body.lstrip("\r\n")), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return info, df


def pick_moment_column(df, requested=None):
    if requested:
        if requested not in df.columns:
            raise ValueError(f"column '{requested}' not present in file")
        return requested
    for c in MOMENT_PREFERENCE:
        if c in df.columns and df[c].notna().any():
            return c
    raise ValueError("no usable moment column found")


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
def parse_filename(path: Path):
    """Pull sample id, mass, measurement type, field / temperature, branch."""
    stem = path.stem
    meta = {"path": path, "stem": stem, "kind": None, "sample": stem,
            "mass_mg": None, "field_oe": None, "temp_k": None, "branch": None}

    m = re.search(r"_(MvT|MvH)(_|$)", stem, flags=re.IGNORECASE)
    if m:
        meta["kind"] = m.group(1).upper().replace("MVT", "MvT").replace("MVH", "MvH")
        meta["sample"] = stem[:m.start()]
        tail = stem[m.end():]
    else:
        tail = ""

    mm = re.search(r"(\d+)p(\d+)mg", stem, flags=re.IGNORECASE)
    if mm:
        meta["mass_mg"] = float(f"{mm.group(1)}.{mm.group(2)}")
    else:
        mm = re.search(r"(\d+(?:\.\d+)?)mg", stem, flags=re.IGNORECASE)
        if mm:
            meta["mass_mg"] = float(mm.group(1))

    fm = re.search(r"(\d+(?:p\d+)?(?:\.\d+)?)\s*(Oe|T)(?![a-zA-Z])", tail, flags=re.IGNORECASE)
    if fm:
        val = float(fm.group(1).replace("p", "."))
        meta["field_oe"] = val * (1e4 if fm.group(2).lower() == "t" else 1.0)

    tm = re.search(r"(\d+(?:p\d+)?(?:\.\d+)?)\s*K(?![a-zA-Z])", tail)
    if tm:
        meta["temp_k"] = float(tm.group(1).replace("p", "."))

    if re.search(r"(^|_)ZFC(_|$)", tail, flags=re.IGNORECASE):
        meta["branch"] = "ZFC"
    elif re.search(r"(^|_)FC(_|$)", tail, flags=re.IGNORECASE):
        meta["branch"] = "FC"
    return meta


def fmt_field(field_oe, mode="auto"):
    if field_oe is None:
        return "?"
    if mode == "t" or (mode == "auto" and field_oe >= 10000):
        return f"{field_oe / 1e4:g} T"
    return f"{field_oe:g} Oe"


def field_tag(field_oe):
    return "unknownField" if field_oe is None else f"{field_oe:g}Oe"


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
# Units are typeset upright (\mathrm); quantity symbols (chi, M, T, H) stay italic.
MVT_NORMS = {
    "emu": (r"$\mathrm{emu\ Oe^{-1}}$", "emu/Oe"),
    "g":   (r"$\mathrm{emu\ Oe^{-1}\ g^{-1}}$", "emu/Oe/g"),
    "fu":  (r"$\mathrm{emu\ Oe^{-1}\ mol_{f.u.}^{-1}}$", "emu/Oe/mol_f.u."),
    "ion": (r"$\mathrm{emu\ Oe^{-1}\ mol_{%s}^{-1}}$", "emu/Oe/mol_%s"),
}
MVT_INV_UNITS = {
    "emu": r"$\mathrm{Oe\ emu^{-1}}$",
    "g":   r"$\mathrm{Oe\ g\ emu^{-1}}$",
    "fu":  r"$\mathrm{Oe\ mol_{f.u.}\ emu^{-1}}$",
    "ion": r"$\mathrm{Oe\ mol_{%s}\ emu^{-1}}$",
}
MVH_NORMS = {
    "emu":     (r"$M$ ($\mathrm{emu}$)", "emu"),
    "g":       (r"$M$ ($\mathrm{emu\ g^{-1}}$)", "emu/g"),
    "fu":      (r"$M$ ($\mathrm{emu\ mol_{f.u.}^{-1}}$)", "emu/mol_f.u."),
    "ion":     (r"$M$ ($\mathrm{emu\ mol_{%s}^{-1}}$)", "emu/mol_%s"),
    "mub_fu":  (r"$M$ ($\mu_\mathrm{B}$ / f.u.)", "muB/f.u."),
    "mub_ion": (r"$M$ ($\mu_\mathrm{B}$ / %s)", "muB/%s"),
}


def norm_factor(kind, norm, mass_g, mw, n_mag, ion):
    """Multiplier converting raw emu (or emu/Oe) into the requested unit."""
    mol_fu = mass_g / mw if (mass_g and mw) else None
    if norm == "emu":
        return 1.0
    if norm == "g":
        return 1.0 / mass_g
    if norm in ("fu", "mub_fu"):
        f = 1.0 / mol_fu
        return f / EMU_PER_MOL_PER_MUB if norm == "mub_fu" else f
    if norm in ("ion", "mub_ion"):
        if not n_mag:
            raise ValueError("magnetic ion count is zero -- cannot normalise per ion")
        f = 1.0 / (mol_fu * n_mag)
        return f / EMU_PER_MOL_PER_MUB if norm == "mub_ion" else f
    raise ValueError(f"unknown normalisation '{norm}'")


def mvt_axis_labels(norm, ion):
    """Return (chi label, 1/chi label, plain unit string)."""
    tex, plain = MVT_NORMS[norm]
    inv_tex = MVT_INV_UNITS[norm]
    if norm == "ion":
        tex, plain, inv_tex = tex % ion, plain % ion, inv_tex % ion
    return f"$\\chi$ ({tex})", f"$\\chi^{{-1}}$ ({inv_tex})", plain


def mvh_axis_label(norm, ion):
    tex, plain = MVH_NORMS[norm]
    if norm in ("ion", "mub_ion"):
        tex, plain = tex % ion, plain % ion
    return tex, plain


NORM_TAG = {"emu": "PerSample", "g": "PerGram", "fu": "PerUnitFormula",
            "ion": "Per%sIon", "mub_fu": "muB-PerUnitFormula",
            "mub_ion": "muB-Per%sIon"}


def norm_tag(norm, ion):
    t = NORM_TAG[norm]
    return t % ion if "%s" in t else t


# --------------------------------------------------------------------------- #
# Plot styling
# --------------------------------------------------------------------------- #
def make_axes(args):
    """Figure with an axes box of exactly args.figsize inches."""
    w, h = args.figsize
    left, bottom, right, top = 1.15, 0.85, 0.30, 0.25
    fig = plt.figure(figsize=(w + left + right, h + bottom + top))
    hsz = [Size.Fixed(left), Size.Fixed(w)]
    vsz = [Size.Fixed(bottom), Size.Fixed(h)]
    div = Divider(fig, (0.0, 0.0, 1.0, 1.0), hsz, vsz, aspect=False)
    ax = fig.add_axes(div.get_position(), axes_locator=div.new_locator(nx=1, ny=1))
    style_axes(ax, args)
    return fig, ax


def style_axes(ax, args):
    fs = args.fontsize
    lw = args.axeslw
    for s in ax.spines.values():
        s.set_linewidth(lw)
    ax.tick_params(which="major", direction="in", length=args.major_tick,
                   width=lw, labelsize=fs, top=True, right=True, bottom=True, left=True)
    ax.tick_params(which="minor", direction="in", length=args.minor_tick,
                   width=lw, top=True, right=True, bottom=True, left=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    steps = [1, 2, 2.5, 5, 10]
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, steps=steps))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, steps=steps))
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 3), useMathText=True)
    ax.xaxis.get_offset_text().set_fontsize(fs - 3)
    ax.yaxis.get_offset_text().set_fontsize(fs - 3)


def finish(ax, args, legend=True):
    if legend and ax.get_legend_handles_labels()[0]:
        ms_min = min(getattr(ax, "_ms_used", [4.0]) or [4.0])
        scale = args.legend_markerscale or float(np.clip(5.0 / ms_min, 1.0, 4.0))
        leg = ax.legend(fontsize=args.legend_fontsize, markerscale=scale, frameon=True,
                        facecolor="white", edgecolor="black", framealpha=1.0,
                        borderpad=0.35, handletextpad=0.5, labelspacing=0.3,
                        loc=args.legend_loc)
        leg.get_frame().set_linewidth(args.axeslw)
        leg.set_zorder(10)
    ax.xaxis.get_offset_text().set_fontsize(args.fontsize - 3)
    ax.yaxis.get_offset_text().set_fontsize(args.fontsize - 3)


def colours(n, cmap_name):
    cmap = plt.get_cmap(cmap_name)
    if n == 1:
        return [cmap(0.15)]
    return [cmap(v) for v in np.linspace(0.0, 0.75, n)]


def ms_for(arr, args):
    """Shrink markers automatically for dense sweeps unless the user set a size."""
    if args.markersize is not None:
        return args.markersize
    n = len(arr)
    return 4.0 if n <= 400 else (2.5 if n <= 1500 else 1.5)


def plot_series(ax, x, y, args, colour, label, filled=True, zorder=3,
                markersize=None, style="markers"):
    """filled=True -> filled markers / solid line;  False -> open markers / dashed."""
    if style == "line":
        ax.plot(x, y, color=colour, lw=args.linewidth,
                ls="-" if filled else (0, (6, 3)), label=label, zorder=zorder)
    else:
        ms = markersize if markersize is not None else ms_for(x, args)
        getattr(ax, "_ms_used", ax.__dict__.setdefault("_ms_used", [])).append(ms)
        ax.plot(x, y, marker=args.marker, ms=ms, color=colour,
                mfc=colour if filled else "none", mec=colour,
                mew=min(args.axeslw, max(ms / 4.0, 0.4)),
                ls="-" if style == "both" else "none", lw=args.linewidth,
                label=label, zorder=zorder)


def save(fig, path, args):
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}")


# --------------------------------------------------------------------------- #
# Curie-Weiss fitting
# --------------------------------------------------------------------------- #
def _stats(y, yfit, npar):
    resid = y - yfit
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    dof = max(len(y) - npar, 1)
    return {"R2": r2, "RMSE": float(np.sqrt(ss_res / len(y))),
            "chi2_red": ss_res / dof, "N": int(len(y))}


def fit_linear_invchi(T, chi):
    """Straight line to 1/chi = (T - theta)/C."""
    inv = 1.0 / chi
    slope, intercept = np.polyfit(T, inv, 1)
    C = 1.0 / slope
    theta = -intercept * C
    res = _stats(inv, slope * T + intercept, 2)
    n = len(T)
    resid = inv - (slope * T + intercept)
    s_err = np.sqrt(np.sum(resid ** 2) / max(n - 2, 1))
    Sxx = np.sum((T - T.mean()) ** 2)
    d_slope = s_err / np.sqrt(Sxx) if Sxx > 0 else np.nan
    d_int = s_err * np.sqrt(1.0 / n + T.mean() ** 2 / Sxx) if Sxx > 0 else np.nan
    dC = abs(C) * (d_slope / abs(slope)) if slope else np.nan
    dtheta = abs(theta) * np.sqrt((d_int / intercept) ** 2 + (d_slope / slope) ** 2) \
        if intercept and slope else np.nan
    res.update({"model": "linear fit to 1/chi", "C": C, "dC": dC,
                "theta": theta, "dtheta": dtheta, "chi0": 0.0, "dchi0": 0.0})
    return res


def fit_cw(T, chi, modified):
    def f_cw(t, C, th):
        return C / (t - th)

    def f_mcw(t, C, th, c0):
        return c0 + C / (t - th)

    lin = fit_linear_invchi(T, chi)
    th_max = float(np.min(T)) - 0.01           # keep the pole outside the fit window
    th0 = min(lin["theta"], th_max - 1.0)
    p0 = [max(lin["C"], 1e-12), th0] + ([0.0] if modified else [])
    fn = f_mcw if modified else f_cw
    lo = [0.0, -np.inf, -np.inf][: len(p0)]
    hi = [np.inf, th_max, np.inf][: len(p0)]
    popt, pcov = curve_fit(fn, T, chi, p0=p0, bounds=(lo, hi), maxfev=200000)
    perr = np.sqrt(np.diag(pcov))
    res = _stats(chi, fn(T, *popt), len(popt))
    res.update({
        "model": "modified Curie-Weiss (NLLS on chi)" if modified
                 else "Curie-Weiss (NLLS on chi)",
        "C": popt[0], "dC": perr[0], "theta": popt[1], "dtheta": perr[1],
        "chi0": popt[2] if modified else 0.0,
        "dchi0": perr[2] if modified else 0.0,
    })
    if abs(popt[1] - th_max) < 1e-3:
        res["warning"] = ("theta reached its upper bound (just below the fit window); "
                          "the fit window probably sits too close to the ordering "
                          "temperature")
    return res


def mu_eff(C):
    return np.sqrt(8.0 * C) if C > 0 else float("nan")


def eval_fit(res, T):
    return res["chi0"] + res["C"] / (T - res["theta"])


# --------------------------------------------------------------------------- #
# Text output helpers
# --------------------------------------------------------------------------- #
def write_columns(path, meta_lines, columns):
    """columns = list of (header, 1-D array); ragged lengths are padded blank."""
    n = max(len(c[1]) for c in columns)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for line in meta_lines:
            fh.write(f"# {line}\n" if line else "#\n")
        fh.write("\t".join(h for h, _ in columns) + "\n")
        for i in range(n):
            row = []
            for _, arr in columns:
                row.append(f"{arr[i]:.8g}" if i < len(arr) and np.isfinite(arr[i]) else "")
            fh.write("\t".join(row) + "\n")
    print(f"  wrote {path.name}")


def provenance(sample, args, formula, mw, n_mag, ion, extra=None):
    lines = [
        f"Generated by plot_mpms.py v{__version__}",
        f"Sample id      : {sample}",
        f"Formula unit   : {formula}   (M = {mw:.4f} g/mol)",
        f"Magnetic ion   : {ion}   ({n_mag:g} per formula unit)" if ion else
        "Magnetic ion   : (none specified)",
    ]
    if args.chi_dia_applied is not None:
        lines.append(f"Diamagnetic corr.: chi_dia = {args.chi_dia_applied:.6g} emu/mol f.u. "
                     f"(subtracted)")
    lines += extra or []
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
class Curve:
    """One .dat file, parsed and normalised."""

    def __init__(self, meta, args):
        self.meta = meta
        self.path = meta["path"]
        self.info, df = read_dat(self.path)
        self.mcol = pick_moment_column(df, args.moment_col)
        self.ecol = ERR_FOR_MOMENT.get(self.mcol)

        keep = ["Temperature (K)", "Magnetic Field (Oe)", self.mcol]
        if self.ecol in df.columns:
            keep.append(self.ecol)
        d = df[keep].apply(pd.to_numeric, errors="coerce").dropna(
            subset=["Temperature (K)", "Magnetic Field (Oe)", self.mcol])
        if args.thin > 1:
            d = d.iloc[:: args.thin]
        self.T = d["Temperature (K)"].to_numpy()
        self.H = d["Magnetic Field (Oe)"].to_numpy()
        self.M = d[self.mcol].to_numpy()
        self.dM = d[self.ecol].to_numpy() if self.ecol in d.columns else None

        # mass: header (mg) preferred, filename as cross-check
        hdr_mass = self.info.get("SAMPLE_MASS", "").strip()
        self.mass_header = float(hdr_mass) if hdr_mass else None
        self.mass_file = meta["mass_mg"]
        if args.mass is not None:
            self.mass_mg, self.mass_src = args.mass, "--mass override"
        elif self.mass_header:
            self.mass_mg, self.mass_src = self.mass_header, "file header"
        elif self.mass_file:
            self.mass_mg, self.mass_src = self.mass_file, "filename"
        else:
            raise ValueError(f"{self.path.name}: no sample mass found; use --mass")
        self.mass_warn = (self.mass_header and self.mass_file and
                          abs(self.mass_header - self.mass_file) > 0.005)
        self.material = self.info.get("SAMPLE_MATERIAL", "")

    @property
    def mass_g(self):
        return self.mass_mg * 1e-3

    def chi_raw(self):
        """chi = M/H in emu/Oe, using the mean measured field."""
        Hmean = float(np.mean(self.H))
        if abs(Hmean) < 1e-9:
            raise ValueError(f"{self.path.name}: mean field is zero -- cannot form M/H")
        return self.M / Hmean, Hmean


NICE_FACTORS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def nice_scale(ratio):
    """Largest 1/2/5-decade factor not exceeding `ratio` (1 when ratio < 2)."""
    f = 1
    for v in NICE_FACTORS:
        if v <= ratio:
            f = v
        else:
            break
    return f


def mvh_select_branch(H, M, mode):
    if mode == "all":
        return H, M
    d = np.sign(np.diff(H))
    d = d[d != 0]
    rev = np.where(np.diff(np.sign(np.diff(H))) != 0)[0]
    if len(rev) == 0:
        return H, M
    first = rev[0] + 1
    if mode == "initial":
        return H[: first + 1], M[: first + 1]
    if mode == "loop":
        return H[first:], M[first:]
    return H, M


# --------------------------------------------------------------------------- #
# Interactive selection
# --------------------------------------------------------------------------- #
def parse_selection(text, n):
    out = []
    for chunk in text.replace(" ", ",").split(","):
        if not chunk:
            continue
        if "-" in chunk and not chunk.startswith("-"):
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    bad = [i for i in out if not 1 <= i <= n]
    if bad:
        raise ValueError(f"index out of range: {bad}")
    return sorted(set(out))


def choose_files(folder, args):
    pattern = "**/*.dat" if args.recursive else "*.dat"
    files = sorted(p for p in folder.glob(pattern) if p.is_file())
    if not files:
        sys.exit(f"No .dat files found in {folder}")

    if args.files:
        chosen = [p for p in files
                  if any(fnmatch.fnmatch(p.name, pat) or fnmatch.fnmatch(p.stem, pat)
                         for pat in args.files)]
        if not chosen:
            sys.exit(f"No files matched: {' '.join(args.files)}")
        return chosen
    if args.all:
        return files

    metas = [parse_filename(p) for p in files]
    print(f"\n.dat files in {folder}:\n")
    w = max(len(p.name) for p in files)
    for i, (p, m) in enumerate(zip(files, metas), 1):
        if m["kind"] == "MvT":
            desc = f"MvT  {fmt_field(m['field_oe'], 'oe'):>10}  {m['branch'] or '-'}"
        elif m["kind"] == "MvH":
            desc = f"MvH  {('%g K' % m['temp_k']) if m['temp_k'] else '?':>10}"
        else:
            desc = "(unrecognised)"
        mg = f"{m['mass_mg']:g} mg" if m["mass_mg"] else ""
        print(f"  [{i:>3}] {p.name:<{w}}  {desc:<22} {mg}")
    print("\nSelect files: e.g. '1,3-5,8'   |  'all'  |  'mvt'  |  'mvh'  |  'q' to quit")
    while True:
        raw = input("> ").strip().lower()
        if raw in ("q", "quit", "exit"):
            sys.exit(0)
        if raw == "all":
            return files
        if raw in ("mvt", "mvh"):
            sel = [p for p, m in zip(files, metas)
                   if (m["kind"] or "").lower() == raw]
            if sel:
                return sel
            print("  none of that type; try again")
            continue
        try:
            idx = parse_selection(raw, len(files))
        except Exception as e:
            print(f"  {e}; try again")
            continue
        if idx:
            return [files[i - 1] for i in idx]


def ask(prompt, default=None):
    suffix = f" [{default}]" if default else ""
    while True:
        val = input(f"{prompt}{suffix}: ").strip()
        if val:
            return val
        if default is not None:
            return default


# --------------------------------------------------------------------------- #
# MvT plotting
# --------------------------------------------------------------------------- #
def do_mvt(sample, curves, args, chem, outdir):
    """curves: list of Curve with kind MvT for one sample."""
    formula, mw, n_mag, ion, chi_dia = chem

    groups = OrderedDict()
    for c in sorted(curves, key=lambda c: (c.meta["field_oe"] or 0,
                                           c.meta["branch"] != "ZFC")):
        groups.setdefault(c.meta["field_oe"], []).append(c)

    fields = list(groups.keys())
    cols = dict(zip(fields, colours(len(fields), args.cmap)))

    fits = {}
    if args.fit:
        for f, cl in groups.items():
            for c in cl:
                br = (c.meta["branch"] or "ZFC").upper()
                if args.fit_which != "both" and br != args.fit_which.upper():
                    continue
                fits[c.path] = run_fit(c, args, chem, outdir)
        if not fits:
            print(f"  ! --fit was requested but no {args.fit_which.upper()} curve was "
                  f"selected for {sample}; use --fit-which to change this")

    for norm in args.mvt_norm:
        ylab, ylab_inv, unit = mvt_axis_labels(norm, ion)
        made = []
        for f, cl in groups.items():
            made.append((f"{sample}_MvT_{field_tag(f)}", {f: cl}))
        if len(groups) > 1:
            made.append((f"{sample}_MvT_all-fields", groups))

        for base, grp in made:
            for inverse in (False, True):
                fig, ax = make_axes(args)
                fit_traces = []
                for f, cl in grp.items():
                    for c in cl:
                        chi = normalised_chi(c, norm, chem)
                        y = 1.0 / chi if inverse else chi
                        br = (c.meta["branch"] or "").upper()
                        lab = fmt_field(f, args.field_units)
                        if br:
                            lab += f", {br}"
                        plot_series(ax, c.T, y, args, cols[f], lab,
                                    filled=(br != "ZFC"), style=args.mvt_style,
                                    markersize=ms_for(c.T, args))
                        if fits.get(c.path):
                            r = fits[c.path]
                            Tf = np.linspace(*args.fit_range, 200)
                            chif = eval_fit(r["used"], Tf) * r["scale"][norm]
                            fit_traces.append((Tf, 1.0 / chif if inverse else chif))
                for j, (Tf, yf) in enumerate(fit_traces):
                    ax.plot(Tf, yf, "-", color="k", lw=args.linewidth, zorder=9,
                            label="fit" if j == 0 else None)
                ax.set_xlabel(r"$T$ ($\mathrm{K}$)", fontsize=args.fontsize)
                ax.set_ylabel(ylab_inv if inverse else ylab, fontsize=args.fontsize)
                ax.set_xlim(left=0)
                finish(ax, args)
                tag = "X-1vT" if inverse else "XvT"
                save(fig, outdir / f"{base}_{norm_tag(norm, ion)}_{tag}.png", args)

            if args.export_txt:
                cols_out = []
                for f, cl in grp.items():
                    for c in cl:
                        chi = normalised_chi(c, norm, chem)
                        lbl = f"{field_tag(f)}_{c.meta['branch'] or 'NA'}"
                        cols_out += [(f"T_{lbl} (K)", c.T),
                                     (f"chi_{lbl} ({unit})", chi),
                                     (f"1/chi_{lbl} (1/({unit}))", 1.0 / chi)]
                        if c.path in fits:
                            r = fits[c.path]
                            Tf = np.linspace(*args.fit_range, 200)
                            chif = eval_fit(r["used"], Tf) * r["scale"][norm]
                            cols_out += [(f"Tfit_{lbl} (K)", Tf),
                                         (f"chifit_{lbl} ({unit})", chif),
                                         (f"1/chifit_{lbl} (1/({unit}))", 1.0 / chif)]
                extra = [f"Measurement    : chi = M/H vs T",
                         f"Normalisation  : {unit}"]
                for f, cl in grp.items():
                    for c in cl:
                        extra.append(
                            f"Source         : {c.path.name}  "
                            f"[{fmt_field(f, 'oe')}, {c.meta['branch'] or '-'}, "
                            f"m = {c.mass_mg:g} mg, column '{c.mcol}']")
                write_columns(outdir / f"{base}_{norm_tag(norm, ion)}_XvT_and_X-1vT.txt",
                              provenance(sample, args, formula, mw, n_mag, ion, extra),
                              cols_out)


def normalised_chi(c, norm, chem):
    formula, mw, n_mag, ion, chi_dia = chem
    chi, _ = c.chi_raw()
    if chi_dia:                       # chi_dia given per mol f.u.
        mol_fu = c.mass_g / mw
        chi = chi - chi_dia * mol_fu
    return chi * norm_factor("MvT", norm, c.mass_g, mw, n_mag, ion)


# --------------------------------------------------------------------------- #
# Fitting driver
# --------------------------------------------------------------------------- #
def run_fit(c, args, chem, outdir):
    formula, mw, n_mag, ion, chi_dia = chem
    base_norm = "ion" if (ion and n_mag) else "fu"
    chi = normalised_chi(c, base_norm, chem)
    lo, hi = args.fit_range
    m = (c.T >= lo) & (c.T <= hi) & np.isfinite(chi) & (chi != 0)
    if m.sum() < 5:
        print(f"  ! {c.path.name}: only {m.sum()} points in {lo}-{hi} K -- skipping fit")
        return None
    T, X = c.T[m], chi[m]

    results = OrderedDict()
    results["linear"] = fit_linear_invchi(T, X)
    try:
        results["cw"] = fit_cw(T, X, modified=False)
    except Exception as e:
        results["cw"] = {"model": "Curie-Weiss (NLLS on chi)", "error": str(e)}
    try:
        results["mcw"] = fit_cw(T, X, modified=True)
    except Exception as e:
        results["mcw"] = {"model": "modified Curie-Weiss (NLLS on chi)", "error": str(e)}

    key = {"nlls": {"mcw": "mcw", "cw": "cw"}[args.fit_model],
           "linear": "linear"}[args.fit_method]
    used = results[key]
    if "error" in used:
        print(f"  ! fit failed for {c.path.name}: {used['error']}")
        return None

    # conversion of the fitted chi (per base_norm) into each plotted norm
    f_base = norm_factor("MvT", base_norm, c.mass_g, mw, n_mag, ion)
    scale = {n: norm_factor("MvT", n, c.mass_g, mw, n_mag, ion) / f_base
             for n in MVT_NORMS}

    write_fit_txt(c, args, chem, results, key, base_norm, outdir)
    return {"used": used, "all": results, "scale": scale, "base_norm": base_norm}


def write_fit_txt(c, args, chem, results, key, base_norm, outdir):
    formula, mw, n_mag, ion, chi_dia = chem
    per = f"mol {ion}" if base_norm == "ion" else "mol f.u."
    path = outdir / f"{c.path.stem}_fitted-results.txt"
    Hmean = float(np.mean(c.H))
    lo, hi = args.fit_range
    L = []
    L.append(f"Curie-Weiss analysis -- plot_mpms.py v{__version__}")
    L.append("=" * 70)
    L.append("")
    L.append("SOURCE")
    L.append(f"  Data file            : {c.path.name}")
    L.append(f"  Full path            : {c.path}")
    L.append(f"  MPMS SAMPLE_MATERIAL : {c.material}")
    L.append(f"  File opened          : {c.info.get('FILEOPENTIME','')}")
    L.append(f"  Moment column        : {c.mcol}")
    L.append(f"  Measurement          : MvT, {fmt_field(c.meta['field_oe'],'oe')} nominal, "
             f"{Hmean:.2f} Oe measured mean, {c.meta['branch'] or '-'}")
    L.append(f"  Points in file       : {len(c.T)}   "
             f"T range {c.T.min():.2f} - {c.T.max():.2f} K")
    L.append("")
    L.append("SAMPLE")
    L.append(f"  Mass                 : {c.mass_mg:g} mg   (source: {c.mass_src})")
    L.append(f"  Formula unit         : {formula}")
    L.append(f"  Molar mass           : {mw:.4f} g/mol")
    L.append(f"  Moles of f.u.        : {c.mass_g / mw:.6e} mol")
    if ion:
        L.append(f"  Magnetic ion         : {ion}, {n_mag:g} per f.u. "
                 f"({c.mass_g / mw * n_mag:.6e} mol)")
    L.append(f"  Diamagnetic corr.    : "
             + (f"{chi_dia:.6e} emu/mol f.u. subtracted" if chi_dia else "none applied"))
    L.append("")
    L.append("FIT SETUP")
    L.append(f"  Fit range            : {lo:g} - {hi:g} K")
    L.append(f"  chi normalised per   : {per}")
    L.append(f"  Reported model       : {results[key]['model']}   <-- used for the figure")
    L.append("")
    L.append("RESULTS  (all three analyses reported for comparison)")
    for k, r in results.items():
        L.append("")
        L.append(f"  [{k}] {r['model']}")
        if "error" in r:
            L.append(f"       FAILED: {r['error']}")
            continue
        star = "   <-- plotted" if k == key else ""
        L.append(f"       C        = {r['C']:.6e} +/- {r['dC']:.2e} "
                 f"emu K Oe^-1 ({per})^-1{star}")
        L.append(f"       theta    = {r['theta']:.4f} +/- {r['dtheta']:.4f} K")
        if r["chi0"] or k == "mcw":
            L.append(f"       chi0     = {r['chi0']:.6e} +/- {r['dchi0']:.2e} "
                     f"emu Oe^-1 ({per})^-1")
        L.append(f"       mu_eff   = {mu_eff(r['C']):.4f} muB per "
                 f"{ion if base_norm == 'ion' else 'f.u.'}")
        if base_norm == "ion":
            L.append(f"       C (f.u.) = {r['C'] * n_mag:.6e} emu K Oe^-1 mol_f.u.^-1")
            L.append(f"       mu_eff (per f.u.) = {mu_eff(r['C'] * n_mag):.4f} muB")
        resid_of = "1/chi" if k == "linear" else "chi"
        L.append(f"       N points = {r['N']}   R^2 = {r['R2']:.6f}   "
                 f"RMSE = {r['RMSE']:.4e} (residuals of {resid_of})   "
                 f"chi2_red = {r['chi2_red']:.4e}")
        if r.get("warning"):
            L.append(f"       WARNING: {r['warning']}")
    L.append("")
    L.append("NOTES")
    L.append("  mu_eff = sqrt(8 C) with C in emu K Oe^-1 mol^-1.")
    L.append("  theta < 0 indicates net antiferromagnetic correlations.")
    L.append("  'linear' fits a straight line to 1/chi(T) and is sensitive to any")
    L.append("  temperature-independent term; 'mcw' absorbs that term into chi0.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  wrote {path.name}")


# --------------------------------------------------------------------------- #
# MvH plotting
# --------------------------------------------------------------------------- #
def do_mvh(sample, curves, args, chem, outdir):
    formula, mw, n_mag, ion, chi_dia = chem
    curves = sorted(curves, key=lambda c: (c.meta["temp_k"] if c.meta["temp_k"]
                                           is not None else np.mean(c.T)))
    cols = colours(len(curves), args.cmap)
    xscale = 1e-4 if args.h_units == "t" else 1.0
    xlabel = (r"$\mu_0 H$ ($\mathrm{T}$)" if args.h_units == "t"
              else r"$H$ ($\mathrm{Oe}$)")

    # branch selection and, optionally, a nice-number scale factor so that a weak
    # high-temperature loop is visible next to a strong low-temperature one
    prepared = []
    for c in curves:
        H, M = mvh_select_branch(c.H, c.M, args.mvh_branch)
        prepared.append([c, H, M, 1])
    ref = max(float(np.max(np.abs(M))) for _, _, M, _ in prepared)
    if args.mvh_scale == "auto":
        for row in prepared:
            mx = float(np.max(np.abs(row[2])))
            if mx > 0:
                row[3] = nice_scale(ref / mx)

    for norm in args.mvh_norm:
        ylab, unit = mvh_axis_label(norm, ion)
        fig, ax = make_axes(args)
        out_cols = []
        extra = [f"Measurement    : M vs H", f"Normalisation  : {unit}"]
        if any(row[3] != 1 for row in prepared):
            extra.append("Scale factors  : some curves are multiplied so that they are "
                         "visible on a common axis; the factor is in the column name "
                         "and in the figure legend")
        for (c, H, M, fac), col in zip(prepared, cols):
            y = M * norm_factor("MvH", norm, c.mass_g, mw, n_mag, ion) * fac
            T = c.meta["temp_k"] if c.meta["temp_k"] is not None else float(np.mean(c.T))
            lab = f"{T:g} K" + (f" ($\\times${fac:g})" if fac != 1 else "")
            tag = f"{T:g}K" + (f"_x{fac:g}" if fac != 1 else "")
            plot_series(ax, H * xscale, y, args, col, lab, filled=True,
                        style=args.mvh_style, markersize=ms_for(H, args))
            out_cols += [(f"H_{tag} ({'T' if args.h_units == 't' else 'Oe'})", H * xscale),
                         (f"M_{tag} ({unit})", y)]
            extra.append(f"Source         : {c.path.name}  [{T:g} K, "
                         f"m = {c.mass_mg:g} mg, column '{c.mcol}', "
                         f"branch '{args.mvh_branch}', scale x{fac:g}]")
        ax.set_xlabel(xlabel, fontsize=args.fontsize)
        ax.set_ylabel(ylab, fontsize=args.fontsize)
        ax.axhline(0, color="0.6", lw=args.axeslw * 0.6, zorder=1)
        ax.axvline(0, color="0.6", lw=args.axeslw * 0.6, zorder=1)
        finish(ax, args)
        base = f"{sample}_MvH_{norm_tag(norm, ion)}"
        save(fig, outdir / f"{base}.png", args)
        if args.export_txt:
            write_columns(outdir / f"{base}.txt",
                          provenance(sample, args, formula, mw, n_mag, ion, extra),
                          out_cols)


# --------------------------------------------------------------------------- #
# Chemistry / mass confirmation
# --------------------------------------------------------------------------- #
def resolve_chemistry(sample, curves, args, cache):
    print(f"\n--- Sample: {sample} ---")
    mismatched = set()
    for c in curves:
        flag = "  <-- differs from filename!" if c.mass_warn else ""
        print(f"  {c.path.name}")
        print(f"      mass {c.mass_mg:g} mg (from {c.mass_src}"
              + (f"; filename says {c.mass_file:g} mg" if c.mass_warn else "")
              + f"){flag}")
        if c.material and sample not in c.material:
            mismatched.add(c.material)
    for mat in sorted(mismatched):
        print(f"  note: header SAMPLE_MATERIAL = '{mat}' does not match the filename "
              f"(mass is still taken from the header)")
    masses = {round(c.mass_mg, 6) for c in curves}
    if len(masses) > 1:
        print(f"  ! WARNING: selected files report different masses: "
              f"{', '.join(f'{m:g}' for m in sorted(masses))} mg")

    cached = cache.get(sample, {})
    formula = args.formula or cached.get("formula")
    ion = args.ion or cached.get("ion")
    if not formula and args.yes:
        sys.exit(f"No formula for sample '{sample}': pass --formula, or drop -y "
                 f"and enter it when prompted.")
    if not formula:
        print("\n" + FORMULA_HELP)
        while True:
            try:
                formula = ask("  Formula unit", cached.get("formula"))
                parse_formula(formula)
                break
            except Exception as e:
                print(f"    {e}")
    counts = parse_formula(formula)
    mw = molar_mass(counts)
    print(f"\n  Formula unit : {format_formula(counts)}")
    for el, n in counts.items():
        print(f"     {el:<2} x {n:<6g} x {ATOMIC_WEIGHTS[el]:>10.5f} = "
              f"{ATOMIC_WEIGHTS[el] * n:>10.4f}")
    print(f"  Molar mass   : {mw:.4f} g/mol")

    if ion is None and not args.no_ion:
        ion = ask("  Magnetic ion (element symbol, blank for none)",
                  cached.get("ion", "")) or None
    if ion and ion not in counts:
        print(f"  ! '{ion}' is not in {formula} -- ignoring")
        ion = None
    n_mag = 0.0
    if ion:
        n_mag = args.n_mag if args.n_mag is not None else counts[ion]
        print(f"  Magnetic ion : {ion}, {n_mag:g} per formula unit")

    chi_dia = 0.0
    if args.chi_dia is not None:
        chi_dia = args.chi_dia
        print(f"  chi_dia      : {chi_dia:.4e} emu/mol f.u. (user supplied)")
    elif args.diamagnetic:
        chi_dia = pascal_chi_dia(counts)
        print(f"  chi_dia      : {chi_dia:.4e} emu/mol f.u. "
              f"(Pascal estimate -- approximate)")
    args.chi_dia_applied = chi_dia or None

    cache[sample] = {"formula": formula, "ion": ion}
    if not args.yes:
        if ask("  Proceed with these values? (y/n)", "y").lower().startswith("n"):
            sys.exit("Aborted.")
    return format_formula(counts), mw, n_mag, ion, chi_dia


def options_walkthrough(args):
    """Offer the two off-by-default extras when the user gave no flags at all."""
    if args.yes:
        return
    if not args.fit:
        args.fit = ask("\nFit the high-temperature MvT data? (y/n)", "n"
                       ).lower().startswith("y")
        if args.fit:
            rng = ask(f"  Fit range in K, as 'LOW HIGH'",
                      f"{args.fit_range[0]:g} {args.fit_range[1]:g}")
            try:
                lo, hi = (float(v) for v in rng.replace(",", " ").split())
                args.fit_range = [lo, hi]
            except Exception:
                print(f"  could not read '{rng}' -- keeping "
                      f"{args.fit_range[0]:g}-{args.fit_range[1]:g} K")
    if not args.export_txt:
        args.export_txt = ask("Also write re-plottable .txt files? (y/n)", "n"
                              ).lower().startswith("y")


def fit_walkthrough(args):
    if not args.fit or args.yes:
        args.fit_model = args.fit_model or "mcw"
        args.fit_method = args.fit_method or "nlls"
        args.fit_which = args.fit_which or "zfc"
        return
    print("\n--- Curie-Weiss fit setup ---")
    print(f"  Fit range: {args.fit_range[0]:g} - {args.fit_range[1]:g} K "
          f"(change with --fit-range LOW HIGH)")
    if args.fit_method is None or args.fit_model is None:
        print("""
  How should the high-temperature data be fitted?
    1  modified Curie-Weiss,  chi = chi0 + C/(T - theta),  non-linear fit to chi(T)
       [default] chi0 absorbs Pauli / Van Vleck / core terms, so C and theta are
       not distorted by them.
    2  plain Curie-Weiss,     chi = C/(T - theta),         non-linear fit to chi(T)
       Use when you are confident there is no temperature-independent term.
    3  straight line to 1/chi(T)
       The traditional construction. Fast and transparent, but a temperature-
       independent term will bend 1/chi and bias theta.
  (All three are computed and written to the results file whatever you pick;
   this choice only sets which one is drawn on the figure and flagged as used.)""")
        choice = ask("  Choice 1/2/3", "1")
        args.fit_model, args.fit_method = {
            "1": ("mcw", "nlls"), "2": ("cw", "nlls"), "3": ("mcw", "linear"),
        }.get(choice, ("mcw", "nlls"))
    if args.fit_which is None:
        print("\n  Which branch(es) should be fitted?")
        print("    1  ZFC only [default]   2  FC only   3  both")
        args.fit_which = {"1": "zfc", "2": "fc", "3": "both"}.get(
            ask("  Choice 1/2/3", "1"), "zfc")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Plot and fit MPMS3 .dat magnetometry data.",
        epilog=FORMULA_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path, nargs="?", default=Path("."),
                   help="folder containing the .dat files (default: current folder)")

    g = p.add_argument_group("file selection")
    g.add_argument("--files", nargs="+", metavar="PATTERN",
                   help="skip the menu; select by glob, e.g. \"NaRhO2*MvH*\"")
    g.add_argument("--all", action="store_true", help="select every .dat in the folder")
    g.add_argument("--recursive", action="store_true", help="search subfolders too")
    g.add_argument("--branch", choices=["both", "fc", "zfc"], default="both",
                   help="which MvT branches to plot (default: both)")
    g.add_argument("--moment-col", metavar="NAME",
                   help="force a moment column (default: auto-detect)")

    g = p.add_argument_group("sample")
    g.add_argument("--formula", metavar="STR", help="formula unit, e.g. Na0.5RhO2")
    g.add_argument("--ion", metavar="EL", help="magnetic ion, e.g. Rh")
    g.add_argument("--no-ion", action="store_true", help="do not ask for a magnetic ion")
    g.add_argument("--n-mag", type=float, metavar="N",
                   help="magnetic ions per f.u. (default: its stoichiometric coefficient)")
    g.add_argument("--mass", type=float, metavar="MG",
                   help="override the sample mass in mg")

    g = p.add_argument_group("normalisation")
    g.add_argument("--mvt-norm", nargs="+", default=["fu"],
                   choices=list(MVT_NORMS), metavar="NORM",
                   help="chi axis: emu | g | fu | ion  (default: fu; several allowed)")
    g.add_argument("--mvh-norm", nargs="+", default=["mub_fu"],
                   choices=list(MVH_NORMS), metavar="NORM",
                   help="M axis: emu | g | fu | ion | mub_fu | mub_ion "
                        "(default: mub_fu; several allowed)")
    g.add_argument("--diamagnetic", action="store_true",
                   help="subtract a Pascal's-constants diamagnetic estimate (default: off)")
    g.add_argument("--chi-dia", type=float, metavar="VALUE",
                   help="explicit diamagnetic chi in emu/mol f.u. (overrides --diamagnetic)")

    g = p.add_argument_group("fitting")
    g.add_argument("--fit", action="store_true",
                   help="Curie-Weiss fit of the high-T MvT data (default: off)")
    g.add_argument("--fit-range", nargs=2, type=float, default=[200.0, 300.0],
                   metavar=("LOW", "HIGH"), help="fit window in K (default: 200 300)")
    g.add_argument("--fit-model", choices=["mcw", "cw"], default=None,
                   help="mcw = chi0 + C/(T-theta) [default], cw = C/(T-theta)")
    g.add_argument("--fit-method", choices=["nlls", "linear"], default=None,
                   help="nlls on chi(T) [default] or straight line to 1/chi")
    g.add_argument("--fit-which", choices=["zfc", "fc", "both"], default=None,
                   help="which branch to fit (default: zfc)")

    g = p.add_argument_group("output")
    g.add_argument("--export-txt", action="store_true",
                   help="also write re-plottable .txt files (default: off)")
    g.add_argument("--outdir", type=Path, default=None,
                   help="output folder (default: alongside the data)")
    g.add_argument("--dpi", type=int, default=300)
    g.add_argument("--no-cache", action="store_true",
                   help=f"do not read or write {CONFIG_NAME} in the data folder")
    g.add_argument("-y", "--yes", action="store_true",
                   help="accept all defaults, never prompt")

    g = p.add_argument_group("style")
    g.add_argument("--figsize", nargs=2, type=float, default=[3.0, 3.0],
                   metavar=("W", "H"), help="AXES size in inches (default: 3 3)")
    g.add_argument("--fontsize", type=float, default=15)
    g.add_argument("--legend-fontsize", type=float, default=None)
    g.add_argument("--legend-loc", default="best")
    g.add_argument("--legend-markerscale", type=float, default=None,
                   help="enlarge legend markers (default: auto)")
    g.add_argument("--major-tick", type=float, default=10)
    g.add_argument("--minor-tick", type=float, default=5)
    g.add_argument("--axeslw", type=float, default=1.0, help="axis/frame width in pt")
    g.add_argument("--cmap", default="viridis")
    g.add_argument("--style", choices=["markers", "line", "both"], default=None,
                   help="set both --mvt-style and --mvh-style at once")
    g.add_argument("--mvt-style", choices=["markers", "line", "both"], default="line",
                   help="chi vs T: FC solid / ZFC dashed by default")
    g.add_argument("--mvh-style", choices=["markers", "line", "both"], default="markers",
                   help="M vs H (default: markers)")
    g.add_argument("--marker", default="o")
    g.add_argument("--markersize", type=float, default=None,
                   help="marker size in pt (default: auto, shrinks for dense sweeps)")
    g.add_argument("--linewidth", type=float, default=2.0,
                   help="curve width in pt for line plots and fit traces (default: 2)")
    g.add_argument("--thin", type=int, default=1, metavar="N",
                   help="plot every Nth point (default: 1)")
    g.add_argument("--h-units", choices=["oe", "t"], default="t",
                   help="MvH x-axis units (default: t)")
    g.add_argument("--field-units", choices=["auto", "oe", "t"], default="auto",
                   help="how fields are written in legends (default: auto)")
    g.add_argument("--mvh-scale", choices=["auto", "none"], default="auto",
                   help="multiply weak MvH curves by a 1/2/5 factor so they are visible "
                        "next to strong ones; the factor is shown in the legend "
                        "(default: auto)")
    g.add_argument("--mvh-branch", choices=["all", "loop", "initial"], default="all",
                   help="all points, the loop after the virgin branch, or only the "
                        "virgin branch (default: all)")

    p.add_argument("--version", action="version", version=f"plot_mpms.py {__version__}")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.chi_dia_applied = None
    if args.style:
        args.mvt_style = args.mvh_style = args.style
    if args.legend_fontsize is None:
        args.legend_fontsize = args.fontsize
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        sys.exit(f"Not a folder: {folder}")
    outdir = (args.outdir or folder).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": args.fontsize, "axes.linewidth": args.axeslw,
                         "mathtext.default": "it", "savefig.facecolor": "white"})

    paths = choose_files(folder, args)
    options_walkthrough(args)
    fit_walkthrough(args)

    cache_path = folder / CONFIG_NAME
    cache = {}
    if cache_path.exists() and not args.no_cache:
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    # load, group by sample then by measurement type
    samples = OrderedDict()
    for p in paths:
        meta = parse_filename(p)
        if meta["kind"] is None:
            print(f"  ! skipping {p.name}: no _MvT_ or _MvH_ token in the name")
            continue
        if meta["kind"] == "MvT" and args.branch != "both":
            if (meta["branch"] or "").upper() != args.branch.upper():
                continue
        try:
            c = Curve(meta, args)
        except Exception as e:
            print(f"  ! skipping {p.name}: {e}")
            continue
        samples.setdefault(meta["sample"], {"MvT": [], "MvH": []})[meta["kind"]].append(c)

    if not samples:
        sys.exit("Nothing to plot.")

    for sample, kinds in samples.items():
        allc = kinds["MvT"] + kinds["MvH"]
        chem = resolve_chemistry(sample, allc, args, cache)
        print()
        if kinds["MvT"]:
            do_mvt(sample, kinds["MvT"], args, chem, outdir)
        if kinds["MvH"]:
            do_mvh(sample, kinds["MvH"], args, chem, outdir)

    if not args.no_cache:
        try:
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except Exception:
            pass
    print(f"\nDone. Output in {outdir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
