#!/usr/bin/env python3
"""
Build a NeurIPS-style PDF from the zero-gap markdown report.

- Parses the Markdown line-by-line.
- Emits LaTeX targeting the official NeurIPS 2024 style file.
- Handles: headings, paragraphs, fenced code, inline code, bold/italic,
  pipe tables, lists, math (\\(...\\) and \\[...\\]), special-char escaping.
- Wide tables are wrapped in \\resizebox for column-fit.
- Long tables get a \\small or \\scriptsize wrapper based on column count.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_MD = ROOT / "mimic_iv_zero_gap_inference_report.md"
OUT_TEX = ROOT / "mimic_iv_zero_gap_inference_neurips.tex"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LATEX_SPECIALS = [
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
]

# Unicode → LaTeX map. Applied BEFORE escape_latex so the substitutions
# survive (they contain backslashes that must not be re-escaped).
UNICODE_TO_LATEX = {
    "≈": r"\ensuremath{\approx}",
    "×": r"\ensuremath{\times}",
    "÷": r"\ensuremath{\div}",
    "±": r"\ensuremath{\pm}",
    "≤": r"\ensuremath{\leq}",
    "≥": r"\ensuremath{\geq}",
    "≠": r"\ensuremath{\neq}",
    "→": r"\ensuremath{\rightarrow}",
    "←": r"\ensuremath{\leftarrow}",
    "↔": r"\ensuremath{\leftrightarrow}",
    "⇒": r"\ensuremath{\Rightarrow}",
    "⇔": r"\ensuremath{\Leftrightarrow}",
    "∞": r"\ensuremath{\infty}",
    "∂": r"\ensuremath{\partial}",
    "∑": r"\ensuremath{\sum}",
    "∏": r"\ensuremath{\prod}",
    "∫": r"\ensuremath{\int}",
    "∇": r"\ensuremath{\nabla}",
    "∈": r"\ensuremath{\in}",
    "∉": r"\ensuremath{\notin}",
    "⊂": r"\ensuremath{\subset}",
    "⊆": r"\ensuremath{\subseteq}",
    "∪": r"\ensuremath{\cup}",
    "∩": r"\ensuremath{\cap}",
    "·": r"\ensuremath{\cdot}",
    "…": r"\ldots{}",
    "—": r"---",
    "–": r"--",
    "“": r"``",
    "”": r"''",
    "‘": r"`",
    "’": r"'",
    "•": r"\ensuremath{\bullet}",
    "✓": r"\ensuremath{\checkmark}",
    "✗": r"\ensuremath{\times}",
    "°": r"\ensuremath{^\circ}",
    "²": r"\ensuremath{^2}",
    "³": r"\ensuremath{^3}",
    "¹": r"\ensuremath{^1}",
    "α": r"\ensuremath{\alpha}",
    "β": r"\ensuremath{\beta}",
    "γ": r"\ensuremath{\gamma}",
    "δ": r"\ensuremath{\delta}",
    "ε": r"\ensuremath{\epsilon}",
    "ζ": r"\ensuremath{\zeta}",
    "η": r"\ensuremath{\eta}",
    "θ": r"\ensuremath{\theta}",
    "ι": r"\ensuremath{\iota}",
    "κ": r"\ensuremath{\kappa}",
    "λ": r"\ensuremath{\lambda}",
    "μ": r"\ensuremath{\mu}",
    "ν": r"\ensuremath{\nu}",
    "ξ": r"\ensuremath{\xi}",
    "π": r"\ensuremath{\pi}",
    "ρ": r"\ensuremath{\rho}",
    "σ": r"\ensuremath{\sigma}",
    "τ": r"\ensuremath{\tau}",
    "υ": r"\ensuremath{\upsilon}",
    "φ": r"\ensuremath{\phi}",
    "χ": r"\ensuremath{\chi}",
    "ψ": r"\ensuremath{\psi}",
    "ω": r"\ensuremath{\omega}",
    "Α": r"\ensuremath{A}",
    "Β": r"\ensuremath{B}",
    "Γ": r"\ensuremath{\Gamma}",
    "Δ": r"\ensuremath{\Delta}",
    "Ε": r"\ensuremath{E}",
    "Λ": r"\ensuremath{\Lambda}",
    "Σ": r"\ensuremath{\Sigma}",
    "Ω": r"\ensuremath{\Omega}",
    "Φ": r"\ensuremath{\Phi}",
    "Π": r"\ensuremath{\Pi}",
    "Ψ": r"\ensuremath{\Psi}",
    "Θ": r"\ensuremath{\Theta}",
    "Ξ": r"\ensuremath{\Xi}",
    "ℓ": r"\ensuremath{\ell}",
    "ℝ": r"\ensuremath{\mathbb{R}}",
    "ℕ": r"\ensuremath{\mathbb{N}}",
    "ℤ": r"\ensuremath{\mathbb{Z}}",
    "√": r"\ensuremath{\surd}",
    "⊤": r"\ensuremath{\top}",
    "⊥": r"\ensuremath{\bot}",
    "⌈": r"\ensuremath{\lceil}",
    "⌉": r"\ensuremath{\rceil}",
    "⌊": r"\ensuremath{\lfloor}",
    "⌋": r"\ensuremath{\rfloor}",
    "Δ": r"\ensuremath{\Delta}",
    "·": r"\ensuremath{\cdot}",
    "\u00a0": r"~",  # non-breaking space
    "−": r"\ensuremath{-}",
    "⁰": r"\ensuremath{^{0}}",
    "¹": r"\ensuremath{^{1}}",
    "²": r"\ensuremath{^{2}}",
    "³": r"\ensuremath{^{3}}",
    "⁴": r"\ensuremath{^{4}}",
    "⁵": r"\ensuremath{^{5}}",
    "⁶": r"\ensuremath{^{6}}",
    "⁷": r"\ensuremath{^{7}}",
    "⁸": r"\ensuremath{^{8}}",
    "⁹": r"\ensuremath{^{9}}",
    "⁻": r"\ensuremath{^{-}}",
    "⁺": r"\ensuremath{^{+}}",
    "₀": r"\ensuremath{_{0}}",
    "₁": r"\ensuremath{_{1}}",
    "₂": r"\ensuremath{_{2}}",
    "₃": r"\ensuremath{_{3}}",
    "₄": r"\ensuremath{_{4}}",
    "ℂ": r"\ensuremath{\mathbb{C}}",
    "ℚ": r"\ensuremath{\mathbb{Q}}",
    "≡": r"\ensuremath{\equiv}",
    "≅": r"\ensuremath{\cong}",
    "⊕": r"\ensuremath{\oplus}",
    "⊗": r"\ensuremath{\otimes}",
    "∝": r"\ensuremath{\propto}",
    "∀": r"\ensuremath{\forall}",
    "∃": r"\ensuremath{\exists}",
    "ω": r"\ensuremath{\omega}",
    "ı": r"\i{}",
    "ℝ": r"\ensuremath{\mathbb{R}}",
    "→": r"\ensuremath{\rightarrow}",
}


# ASCII-fold mapping for code listings (lstlisting cannot easily handle UTF-8).
UNICODE_TO_ASCII = {
    "≈": "~=",
    "×": "x",
    "÷": "/",
    "±": "+/-",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "→": "->",
    "←": "<-",
    "⇒": "=>",
    "⇔": "<=>",
    "∞": "inf",
    "·": "*",
    "…": "...",
    "—": "--",
    "–": "-",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "•": "-", "✓": "[ok]", "✗": "[x]",
    "°": "deg",
    "²": "^2", "³": "^3", "¹": "^1",
    "⁰": "^0", "⁴": "^4", "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
    "⁻": "^-", "⁺": "^+",
    "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4",
    "−": "-",
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "θ": "theta", "λ": "lambda", "μ": "mu",
    "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "φ": "phi",
    "χ": "chi", "ψ": "psi", "ω": "omega",
    "Δ": "Delta", "Λ": "Lambda", "Σ": "Sigma", "Ω": "Omega",
    "ℓ": "l",
    "\u00a0": " ",
    "─": "-", "│": "|", "└": "+", "├": "+", "┘": "+", "┌": "+", "┐": "+",
    "·": "*",
}


def fold_to_ascii(s: str) -> str:
    """For lstlisting bodies: replace Unicode with ASCII fallbacks."""
    return "".join(UNICODE_TO_ASCII.get(ch, ch) for ch in s)


def replace_unicode(s: str) -> str:
    """Replace known Unicode characters with LaTeX equivalents."""
    out = []
    for ch in s:
        out.append(UNICODE_TO_LATEX.get(ch, ch))
    return "".join(out)

INLINE_CODE_RE = re.compile(r"`([^`]+)`")
BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
CODE_FENCE_RE = re.compile(r"^```([A-Za-z0-9_]*)\s*$")
HR_RE = re.compile(r"^\s*-{3,}\s*$")
ULIST_RE = re.compile(r"^(\s*)[-*]\s+(.*)$")
OLIST_RE = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
TABLE_DIVIDER_RE = re.compile(r"^\|\s*[:-]+(\s*\|\s*[:-]+)+\s*\|\s*$")
MATH_BLOCK_RE = re.compile(r"^\\\[(.*)\\\]\s*$")


def escape_latex(s: str) -> str:
    """Escape LaTeX specials in a plain-text string (no markdown semantics).

    Unicode is mapped to LaTeX commands via UNICODE_TO_LATEX. Backslashes that
    appear in those replacements are then re-escaped, so we splice them in
    after the per-character escape pass.
    """
    placeholders: list[tuple[str, str]] = []

    def stash(token: str) -> str:
        idx = len(placeholders)
        placeholders.append((f"@@UC{idx}@@", token))
        return f"@@UC{idx}@@"

    # First, replace Unicode chars with placeholder tokens carrying LaTeX cmds.
    out_chars = []
    for ch in s:
        repl = UNICODE_TO_LATEX.get(ch)
        if repl is not None:
            out_chars.append(stash(repl))
        else:
            out_chars.append(ch)
    s2 = "".join(out_chars)

    # Now LaTeX-escape the remaining text (placeholders survive — they are
    # alphanumeric and '@' which are not LaTeX specials).
    out = []
    for ch in s2:
        repl = None
        for old, new in LATEX_SPECIALS:
            if ch == old:
                repl = new
                break
        out.append(repl if repl is not None else ch)
    s3 = "".join(out)

    # Finally, restore placeholders.
    for ph, val in placeholders:
        s3 = s3.replace(ph, val)
    return s3


def render_inline(s: str) -> str:
    """Render markdown inline elements (code, bold, italic, links, math) to LaTeX.

    Math segments \\( ... \\) are kept verbatim (already LaTeX). Inline-code
    segments are wrapped in \\texttt{} with their contents LaTeX-escaped.
    Bold/italic are converted; remaining text is LaTeX-escaped.
    """
    # Tokenize protecting (1) inline math \( ... \)  (2) inline code `...`
    placeholders: list[tuple[str, str]] = []

    def stash(token: str) -> str:
        idx = len(placeholders)
        placeholders.append((f"@@PH{idx}@@", token))
        return f"@@PH{idx}@@"

    # Protect math first
    def math_sub(m: re.Match) -> str:
        return stash(m.group(0))

    s = re.sub(r"\\\((.+?)\\\)", math_sub, s)
    s = re.sub(r"\$([^$\n]+)\$", lambda m: stash(m.group(0)), s)

    # Protect inline code
    def code_sub(m: re.Match) -> str:
        body = m.group(1)
        body_escaped = escape_latex(body)
        return stash("\\texttt{" + body_escaped + "}")

    s = INLINE_CODE_RE.sub(code_sub, s)

    # Links
    def link_sub(m: re.Match) -> str:
        text = m.group(1)
        url = m.group(2)
        return stash("\\href{" + url + "}{" + escape_latex(text) + "}")

    s = LINK_RE.sub(link_sub, s)

    # Bold / italic markers — handle BEFORE escaping the rest
    def bold_sub(m: re.Match) -> str:
        return stash("\\textbf{" + render_inline(m.group(1)) + "}")

    def italic_sub(m: re.Match) -> str:
        return stash("\\emph{" + render_inline(m.group(1)) + "}")

    s = BOLD_RE.sub(bold_sub, s)
    s = ITALIC_RE.sub(italic_sub, s)

    # Now escape what remains
    s = escape_latex(s)

    # Restore placeholders
    for ph, val in placeholders:
        s = s.replace(escape_latex(ph), val)
        s = s.replace(ph, val)

    return s


# ---------------------------------------------------------------------------
# Block parsers
# ---------------------------------------------------------------------------

def parse_table(lines: list[str], idx: int) -> tuple[str, int]:
    """Parse a markdown pipe-table starting at lines[idx]; return (latex, new_idx).

    The header row is at lines[idx], divider at lines[idx+1], body follows.
    """
    header_line = lines[idx].strip().strip("|")
    cols = [c.strip() for c in header_line.split("|")]
    n_cols = len(cols)
    rows = []
    j = idx + 2
    while j < len(lines) and lines[j].strip().startswith("|"):
        body_line = lines[j].strip().strip("|")
        cells = [c.strip() for c in body_line.split("|")]
        # pad/truncate to n_cols
        if len(cells) < n_cols:
            cells = cells + [""] * (n_cols - len(cells))
        elif len(cells) > n_cols:
            cells = cells[:n_cols]
        rows.append(cells)
        j += 1

    # Pick column spec.
    col_spec = "l" * n_cols if n_cols <= 4 else "p{" + f"{max(0.6, 14.0/n_cols):.2f}cm" + "}" * 0 + ("l" * n_cols)

    # Use booktabs.
    out: list[str] = []
    # Decide font size by total row width
    total_chars = sum(len(c) for c in cols)
    for row in rows:
        total_chars = max(total_chars, sum(len(c) for c in row))
    if n_cols >= 8 or total_chars > 90:
        font_cmd = "\\scriptsize"
        wrap = True
    elif n_cols >= 6 or total_chars > 70:
        font_cmd = "\\small"
        wrap = True
    else:
        font_cmd = ""
        wrap = False

    out.append("\\begin{table}[H]")
    out.append("  \\centering")
    if font_cmd:
        out.append("  " + font_cmd)
    if wrap:
        out.append("  \\resizebox{\\linewidth}{!}{%")
    out.append("  \\begin{tabular}{" + ("l" * n_cols) + "}")
    out.append("    \\toprule")
    out.append("    " + " & ".join(render_inline(c) for c in cols) + " \\\\")
    out.append("    \\midrule")
    for row in rows:
        out.append("    " + " & ".join(render_inline(c) for c in row) + " \\\\")
    out.append("    \\bottomrule")
    out.append("  \\end{tabular}")
    if wrap:
        out.append("  }")
    out.append("\\end{table}")
    return "\n".join(out), j


def parse_code_block(lines: list[str], idx: int) -> tuple[str, int]:
    """Parse a fenced code block starting at lines[idx]; return (latex, new_idx)."""
    fence = lines[idx].strip()
    m = CODE_FENCE_RE.match(fence)
    lang = (m.group(1) if m else "").lower() if m else ""
    j = idx + 1
    body_lines: list[str] = []
    while j < len(lines):
        if CODE_FENCE_RE.match(lines[j].strip()):
            j += 1
            break
        body_lines.append(lines[j])
        j += 1
    body = "\n".join(body_lines)
    body = fold_to_ascii(body)
    # CSV / text / no-lang -> verbatim block. Python -> lstlisting.
    if lang in {"python", "py"}:
        return ("\\begin{lstlisting}[language=Python]\n" + body + "\n\\end{lstlisting}"), j
    elif lang in {"text", ""}:
        return ("\\begin{lstlisting}[basicstyle=\\scriptsize\\ttfamily,breaklines=true]\n" + body + "\n\\end{lstlisting}"), j
    elif lang in {"csv"}:
        return ("\\begin{lstlisting}[basicstyle=\\tiny\\ttfamily,breaklines=true]\n" + body + "\n\\end{lstlisting}"), j
    else:
        return ("\\begin{lstlisting}[basicstyle=\\scriptsize\\ttfamily,breaklines=true]\n" + body + "\n\\end{lstlisting}"), j


def parse_list(lines: list[str], idx: int) -> tuple[str, int]:
    """Parse a contiguous list (un/ordered) starting at lines[idx]."""
    items: list[str] = []
    j = idx
    is_ordered = bool(OLIST_RE.match(lines[idx]))
    while j < len(lines):
        m_u = ULIST_RE.match(lines[j])
        m_o = OLIST_RE.match(lines[j])
        if not (m_u or m_o):
            # Allow blank lines inside lists; stop at non-blank non-list line.
            if lines[j].strip() == "":
                # Peek: if next non-blank line is still a list item, continue.
                k = j + 1
                while k < len(lines) and lines[k].strip() == "":
                    k += 1
                if k < len(lines) and (ULIST_RE.match(lines[k]) or OLIST_RE.match(lines[k])):
                    j = k
                    continue
            break
        text = (m_u or m_o).group(2 if m_u else 3)
        items.append(render_inline(text))
        j += 1
    env = "enumerate" if is_ordered else "itemize"
    out = ["\\begin{" + env + "}[leftmargin=1.4em,itemsep=2pt,topsep=2pt]"]
    for it in items:
        out.append("  \\item " + it)
    out.append("\\end{" + env + "}")
    return "\n".join(out), j


# ---------------------------------------------------------------------------
# Document conversion
# ---------------------------------------------------------------------------

NEURIPS_PREAMBLE = r"""\documentclass{article}

\PassOptionsToPackage{numbers,sort&compress}{natbib}
\usepackage[preprint]{neurips_2024}

% Core packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{float}
\usepackage{enumitem}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{caption}
\usepackage{longtable}
\usepackage{geometry}

% Slightly widen text for tables — preprint mode preserves aesthetics.
\geometry{margin=1in}

\hypersetup{colorlinks=true, linkcolor=blue!50!black, urlcolor=blue!60!black,
            citecolor=green!50!black, pdfborder={0 0 0}}

\lstset{
  basicstyle=\small\ttfamily,
  breaklines=true,
  frame=single,
  framesep=2pt,
  showstringspaces=false,
  keepspaces=true,
  columns=fullflexible,
  upquote=true,
  language=Python,
  commentstyle=\color{green!50!black},
  keywordstyle=\color{blue!70!black}\bfseries,
  stringstyle=\color{red!60!black},
  numbers=none,
  xleftmargin=1em,
}

\title{Federated Optimization on MIMIC-IV: \\
A Zero-Gap, Self-Critiquing Inference Report on \\
Nine ICU Clients, Four Trade-off Axes, \\
and One Convex Sanity Backbone}

\author{%
  Autonomous Research-Report Agent \\
  \texttt{federated-learning-optimization/} \\
  \And
  Source-of-truth: \texttt{outputs/full\_mimic\_iv\{\_training,\_proposal\_alignment\}/} \\
}

\hypersetup{
  pdftitle={Federated Optimization on MIMIC-IV: A Zero-Gap Inference Report},
  pdfauthor={Autonomous Research-Report Agent},
  pdfsubject={Federated learning, MIMIC-IV, ICU mortality},
  pdfkeywords={federated learning, MIMIC-IV, FedAvg, FedProx, CVaR, LP duality, Dirichlet, sparsity}
}

\begin{document}
\maketitle

\begin{abstract}
We study federated optimization for 24-hour ICU mortality prediction on MIMIC-IV with nine natural ICU clients, 73{,}141 stays, 1{,}021 features, and 11.4\% positive rate. We evaluate FedAvg, FedProx with $\mu \in \{0, 0.001, 0.01, 0.1\}$, CVaR-style aggregation reweighting at $\alpha \in \{0, 0.5, 0.75, 0.9, 0.95\}$, centralized and local-only baselines, hyperparameter search via grid search and differential evolution, LP-duality policy programs with KKT diagnostics, top-$k$ communication compression, a synthetic Dirichlet study at $\beta \in \{0.1, 0.5, 1.0, \infty\}$ with $K=30$, and 1D/2D loss-landscape interpolation. We find: FedProx with $\mu=0.1$ is the strongest single method on AUPRC ($0.6653 \pm 0.0051$), AUROC ($0.9128$), and worst-client recall ($0.500 \pm 0.047$); the convex logistic backbone (2{,}044 parameters) confirms the FedProx-$\mu = 0.1$ win with effect size $+1.03$ on AUPRC; client heterogeneity (Dirichlet $\beta$) is the dominant fairness driver, collapsing worst-client recall to $0$ at $\beta=0.1$ and restoring it to $0.727$ at $\beta=\infty$; updates are $95.8\%$--$97.7\%$ nonzero so top-$k$ truncation is the active compression mechanism (\(\sim\!5\times\) at top-1\% with negligible loss penalty). Differential evolution beats grid search on AUPRC, fairness, and communication given a $6\times$ evaluation budget. Every numeric claim cites a CSV/JSON file under \texttt{outputs/}.
\end{abstract}

\clearpage
\tableofcontents
\clearpage
\listoftables
\clearpage
"""

NEURIPS_POSTAMBLE = r"""

\end{document}
"""


def convert_markdown(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    skipped_title = False

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Top-level title (single '# Title') - skip; provided in maketitle
        if not skipped_title and stripped.startswith("# ") and not stripped.startswith("## "):
            i += 1
            skipped_title = True
            continue

        # Skip italic comment line right after the title
        if skipped_title and stripped.startswith("*") and stripped.endswith("*") and "research-grade" in stripped.lower():
            i += 1
            continue

        # Code fence
        if CODE_FENCE_RE.match(stripped):
            block, i = parse_code_block(lines, i)
            out.append(block)
            out.append("")
            continue

        # Math block \[ ... \]
        if stripped.startswith("\\[") and "\\]" in stripped:
            out.append(stripped)
            out.append("")
            i += 1
            continue

        # Multi-line math block: starts with \[ and ends later with \]
        if stripped.startswith("\\["):
            buf = [stripped]
            i += 1
            while i < n and "\\]" not in lines[i]:
                buf.append(lines[i])
                i += 1
            if i < n:
                buf.append(lines[i])
                i += 1
            out.append("\n".join(buf))
            out.append("")
            continue

        # Horizontal rule -> ignore (NeurIPS prefers section breaks)
        if HR_RE.match(stripped):
            i += 1
            continue

        # Heading
        m = HEADING_RE.match(stripped)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            if level == 1:
                # already-skipped main title; if encountered again render as section
                out.append("\\section*{" + render_inline(title) + "}")
            elif level == 2:
                # Force a clean break before each top-level section.
                out.append("\\clearpage")
                out.append("\\section{" + render_inline(title) + "}")
            elif level == 3:
                out.append("\\subsection{" + render_inline(title) + "}")
            elif level == 4:
                out.append("\\subsubsection{" + render_inline(title) + "}")
            else:
                out.append("\\paragraph{" + render_inline(title) + "}")
            out.append("")
            i += 1
            continue

        # Table: header line followed by divider line of dashes
        if stripped.startswith("|") and i + 1 < n and TABLE_DIVIDER_RE.match(lines[i + 1].strip()):
            block, i = parse_table(lines, i)
            out.append(block)
            out.append("")
            continue

        # Lists
        if ULIST_RE.match(line) or OLIST_RE.match(line):
            block, i = parse_list(lines, i)
            out.append(block)
            out.append("")
            continue

        # Blank line
        if stripped == "":
            out.append("")
            i += 1
            continue

        # Paragraph: gather contiguous non-empty non-special lines
        buf = [line]
        i += 1
        while i < n:
            nxt = lines[i]
            ns = nxt.strip()
            if ns == "":
                break
            if HEADING_RE.match(ns) or CODE_FENCE_RE.match(ns) or HR_RE.match(ns):
                break
            if ns.startswith("|"):
                break
            if ULIST_RE.match(nxt) or OLIST_RE.match(nxt):
                break
            if ns.startswith("\\["):
                break
            buf.append(nxt)
            i += 1
        para = " ".join(b.strip() for b in buf)
        out.append(render_inline(para))
        out.append("")

    return "\n".join(out)


def main() -> int:
    if not SRC_MD.exists():
        print(f"ERROR: source markdown not found at {SRC_MD}", file=sys.stderr)
        return 1
    md = SRC_MD.read_text(encoding="utf-8")
    body_tex = convert_markdown(md)

    full = NEURIPS_PREAMBLE + body_tex + NEURIPS_POSTAMBLE
    OUT_TEX.write_text(full, encoding="utf-8")
    print(f"Wrote {OUT_TEX} ({len(full):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
