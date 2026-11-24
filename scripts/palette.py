"""Single source of truth for Not-Just-Speedup figure colors and TikZ preamble.

Mirrors the paper's mint-green palette (palette.tex, plot_behavior.py, heat3d_sota.py):
mint green carries the primary / positive end, burnt orange and peach carry errors and
warnings, muted blue and neutral greys carry secondary/baseline series, and pure black
is used for axes, rules, and text.
"""

# Core theme palette (6-digit hex, no leading '#')
THEME_DARK = "1E6F54"   # primary dark mint (positive / strong / reproduced / high agreement)
THEME_MAIN = "3EB489"   # primary mint accent
THEME_MID = "7BCFAE"    # secondary mint tint
THEME_LIGHT = "B5EAD7"  # light mint highlight

# Semantic accents matched to the paper's existing figures
AMBER = "C2410C"        # computation failure / algorithm change (heat3d_sota.py)
PEACH = "E8A87C"        # engineering failure / debug / went on without gain (plot_behavior.py)
BLUE = "5B8DD6"         # kept borderline engineering / perf. measure (plot_behavior.py)
PURPLE = "8B6FB8"       # secondary categorical accent (plot_behavior.py)
GREY_DARK = "374151"    # muted dark accent
GREY_MID = "9CA3AF"     # neutral baseline / not reproduced (heat3d_sota.py)
GREY_LIGHT = "E5E7EB"   # neutral light / not a bug / none (plot_behavior.py)
WHITE = "FFFFFF"
INK = "000000"

# Entity-to-color mapping (color follows the entity across every figure)
BUG_OUTCOME_COLORS = {
    "reproduced": THEME_DARK,
    "not-reproduced": GREY_MID,
    "not-a-bug": GREY_LIGHT,
    "cannot-test": WHITE,
    "fail_computation": AMBER,
    "fail_engineering": PEACH,
    "kept_computation_borderline": THEME_MID,
    "kept_engineering_borderline": BLUE,
    "waived": GREY_LIGHT,
}

AGREEMENT_BAND_COLORS = {
    "high": THEME_DARK,     # >= 85%
    "mid": THEME_MID,       # 70% - 84%
    "low": GREY_MID,        # < 70% (dominated by "unknown")
}

# Expense ramp (Cost page only): red, light -> dark, monotone in lightness; reserved for list-price cost
EXPENSE_RAMP = ["FFF5F0", "FDD5C5", "F9A487", "EE6548", "C9302A", "8C1515"]
# Token ramps (Cost page): input blue-violet, output plum; neighbouring hues, light -> dark
INPUT_TOKEN_RAMP = ["F5F4FB", "DCDAF0", "B6B1E0", "8A80C9", "6250AD", "3D2C80"]
OUTPUT_TOKEN_RAMP = ["FBF3F8", "F2D1E6", "E3A3CC", "CC6FAD", "A8418A", "74205F"]
# Time ramp (Cost page): blue, light -> dark
TIME_RAMP = ["F2F7FC", "CFE2F3", "9EC4E6", "629DD1", "2F6FB3", "174478"]

TIKZ_DOC = r"""\documentclass[border=3pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[tt=false, type1=true]{libertine}
\usepackage{tikz}
\begin{document}
\fontsize{7}{8.5}\selectfont
\begin{tikzpicture}[x=1pt, y=1pt]
BODY
\end{tikzpicture}
\end{document}
"""


def tikz_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{{rgb,255:red,{int(h[0:2], 16)};green,{int(h[2:4], 16)};blue,{int(h[4:6], 16)}}}"


def tex_escape(text: str) -> str:
    s = str(text)
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]:
        s = s.replace(old, new)
    return s
