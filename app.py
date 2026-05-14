import math
import base64
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# Configuração da página
# =========================================================
st.set_page_config(
    page_title="Simulador Campo Elétrico Cilindro",
    page_icon="⚡",
    layout="wide",
)

# =========================================================
# Constantes
# =========================================================
EPSILON_0 = 8.8e-12  # conforme solicitado no enunciado
L_REF = 1.0          # adotamos 1 m de comprimento de referência


# =========================================================
# Funções utilitárias de formatação
# =========================================================
SUPERSCRIPTS = {
    "-": "⁻",
    "+": "⁺",
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
}

SUBSCRIPTS = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "a": "ₐ",
    "e": "ₑ",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "x": "ₓ",
    "g": "g",
}

def exp_to_superscript(n: int) -> str:
    s = str(n)
    return "".join(SUPERSCRIPTS.get(ch, ch) for ch in s)

def fmt_num(x: float, unit: str = "", digits: int = 4) -> str:
    """
    Formata número sem usar notação e/E.
    Usa forma a × 10ⁿ quando necessário.
    """
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return f"indefinido {unit}".strip()

    if abs(x) < 1e-15:
        return f"0 {unit}".strip()

    ax = abs(x)
    if 1e-3 <= ax < 1e4:
        s = f"{x:.{digits}f}".rstrip("0").rstrip(".")
        return f"{s} {unit}".strip()

    exp = int(math.floor(math.log10(ax)))
    mant = x / (10 ** exp)
    s_mant = f"{mant:.{digits}f}".rstrip("0").rstrip(".")
    return f"{s_mant} × 10{exp_to_superscript(exp)} {unit}".strip()

def fmt_num_html(x: float, unit: str = "", digits: int = 4) -> str:
    """
    Versão HTML com expoente em <sup>.
    """
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return f"indefinido {unit}".strip()

    if abs(x) < 1e-15:
        return f"0 {unit}".strip()

    ax = abs(x)
    if 1e-3 <= ax < 1e4:
        s = f"{x:.{digits}f}".rstrip("0").rstrip(".")
        return f"{s} {unit}".strip()

    exp = int(math.floor(math.log10(ax)))
    mant = x / (10 ** exp)
    s_mant = f"{mant:.{digits}f}".rstrip("0").rstrip(".")
    return f"{s_mant} × 10<sup>{exp}</sup> {unit}".strip()

def charge_color(q: float) -> str:
    if q > 0:
        return "#d62828"  # vermelho
    elif q < 0:
        return "#1d4ed8"  # azul
    return "#111111"      # preto

def soft_fill(hex_color: str, alpha: float = 0.18) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(0,0,0,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def q_gauss(r: float, a: float, b: float, q_total_c: float, is_conductor: bool) -> float:
    """
    Carga contida pela superfície gaussiana, adotando L=1 m.
    """
    if r < 0:
        return 0.0

    # Cavidade interna
    if r < a:
        return 0.0

    # Dentro do material
    if a <= r < b:
        if is_conductor:
            return 0.0
        else:
            denom = (b**2 - a**2)
            if abs(denom) < 1e-15:
                return 0.0
            return q_total_c * (r**2 - a**2) / denom

    # Fora do cilindro
    return q_total_c

def electric_field(r: float, a: float, b: float, q_total_c: float, is_conductor: bool) -> float:
    """
    Campo elétrico assinado (positivo -> radial para fora; negativo -> para dentro)
    em r, para cilindro longo usando L=1 m.
    E = q_gauss / (2*pi*r*L*epsilon0), com L=1 m.
    """
    if r <= 0:
        return 0.0

    qg = q_gauss(r, a, b, q_total_c, is_conductor)
    area = 2 * math.pi * r * L_REF
    return qg / (area * EPSILON_0)

def area_gauss(r: float) -> float:
    return 2 * math.pi * r * L_REF

def rho_volume(q_total_c: float, a: float, b: float) -> float:
    vol = math.pi * (b**2 - a**2) * L_REF
    if abs(vol) < 1e-15:
        return 0.0
    return q_total_c / vol

def maybe_base64_image(path: str):
    p = Path(path)
    if not p.exists():
        return None
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    suffix = p.suffix.lower()
    mime = "image/png"
    if suffix == ".jpg" or suffix == ".jpeg":
        mime = "image/jpeg"
    return f"data:{mime};base64,{b64}"


# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .app-title {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 0.25rem;
        }

        .app-subtitle {
            font-size: 1.05rem;
            color: #374151;
            margin-top: 0.15rem;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 800;
            margin-top: 1.2rem;
            margin-bottom: 0.5rem;
            color: #111827;
        }

        .note-box {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 14px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.75rem;
        }

        .formula-box {
            background: #ffffff;
            border: 1px solid #d1d5db;
            border-left: 5px solid #2563eb;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.7rem;
            overflow-wrap: anywhere;
        }

        .highlight-slider {
            background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%);
            border: 2px solid #60a5fa;
            border-radius: 18px;
            padding: 1rem 1rem 0.5rem 1rem;
            margin-top: 0.5rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.08);
        }

        .small-muted {
            color: #6b7280;
            font-size: 0.95rem;
        }

        .svg-wrap {
            width: 100%;
            overflow-x: auto;
            overflow-y: hidden;
            padding-bottom: 0.35rem;
            border: 1px solid #d1d5db;
            border-radius: 16px;
            background: #ffffff;
        }

        .eq-inline {
            font-size: 1.02rem;
            line-height: 1.7;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Cabeçalho: duas colunas com logo e título
# =========================================================
col_logo, col_title = st.columns([1, 4], vertical_alignment="center")

with col_logo:
    logo_path = Path("logo_maua.png")
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.markdown(
            """
            <div class="note-box" style="text-align:center;">
                <strong>logo_maua.png</strong><br>
                não encontrado
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_title:
    st.markdown('<div class="app-title">Simulador Campo Elétrico Cilindro</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Estude o campo elétrico de um cilindro longo.</div>',
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="note-box">
        Neste simulador, adotamos <strong>L = 1 m</strong> como comprimento de referência do cilindro.
        Como o problema é de um <strong>cilindro longo</strong>, esse comprimento de referência simplifica
        as contas sem alterar a interpretação física do campo elétrico por unidade de comprimento.
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Parâmetros
# =========================================================
st.markdown('<div class="section-title">Parâmetros</div>', unsafe_allow_html=True)

p1, p2, p3 = st.columns([1.1, 1.1, 1.0])

with p1:
    a = st.number_input(
        "Raio interno a do cilindro (m)",
        min_value=0.0,
        max_value=20.0,
        value=0.5,
        step=0.1,
        format="%.2f",
        help="Pode ser zero para cilindro maciço."
    )

with p2:
    b_in = st.number_input(
        "Raio externo b do cilindro (m)",
        min_value=0.5,
        max_value=20.0,
        value=max(1.0, a + 0.5),
        step=0.1,
        format="%.2f",
        help="O app garante b ≥ a + 0,5 m."
    )

with p3:
    q_micro = st.number_input(
        "Carga Q do cilindro (micronC)",
        value=6.0,
        step=1.0,
        format="%.2f",
        help="Digite a carga total do cilindro em microcoulomb."
    )

# Garantia física: b >= a + 0,5
b = max(b_in, a + 0.5)
if b != b_in:
    st.warning(f"O raio externo foi ajustado automaticamente para b = {fmt_num(b, 'm')} para garantir b ≥ a + 0,5 m.")

material_col1, material_col2 = st.columns([1, 3])
with material_col1:
    material = st.radio(
        "Material do cilindro",
        ["Isolante", "Condutor"],
        horizontal=False,
    )

is_conductor = material == "Condutor"
q_total_c = q_micro * 1e-6

r_slider_max = max(2.0, b + 2.0, 1.6 * b + 1.0)

st.markdown('<div class="highlight-slider">', unsafe_allow_html=True)
r = st.slider(
    "Raio da superfície gaussiana r (m) para estudo do campo elétrico",
    min_value=0.0,
    max_value=float(round(r_slider_max, 2)),
    value=float(round(min(max(0.8 * b, 0.1), r_slider_max), 2)),
    step=0.01,
)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# Cálculos principais
# =========================================================
qg = q_gauss(r, a, b, q_total_c, is_conductor)
A_g = area_gauss(r) if r > 0 else 0.0
E_r = electric_field(r, a, b, q_total_c, is_conductor)

# Cargas em superfícies para a figura
if is_conductor:
    q_int_surface = 0.0
    q_ext_surface = q_total_c
else:
    q_int_surface = None
    q_ext_surface = None

# =========================================================
# Seção: Imagem (SVG responsivo com rolagem horizontal)
# =========================================================
st.markdown('<div class="section-title">Imagem</div>', unsafe_allow_html=True)

def build_svg(
    a: float,
    b: float,
    r: float,
    q_total_c: float,
    is_conductor: bool,
    E_r: float,
    qg: float,
    q_int_surface: float = None,
    q_ext_surface: float = None,
):
    # Escala fixa: 1 m -> 85 px (visualização fixa)
    px_per_m = 85.0

    outer_r_px = max(8, a * 0 + b * px_per_m)
    inner_r_px = a * px_per_m
    gauss_r_px = r * px_per_m

    # Geometria fixa do cilindro
    x_front = 240
    x_back = 760
    y_c = 250

    # Largura final do SVG: garante espaço para seta e boxes
    svg_w = int(max(1250, x_back + max(outer_r_px, gauss_r_px) + 350))
    svg_h = int(max(520, y_c + max(outer_r_px, gauss_r_px) + 140))

    # Cores
    color_total = charge_color(q_total_c)
    outer_surface_color = color_total

    if is_conductor:
        inner_surface_color = charge_color(0.0)  # preta
    else:
        inner_surface_color = color_total

    gauss_color = "#16a34a"  # verde
    field_color = "#111827"

    outer_fill = soft_fill(outer_surface_color, 0.20)
    inner_fill = "#ffffff"

    # Arrow direction based on signed field
    arrow_len = 70
    x_arrow_start = x_back + gauss_r_px
    y_arrow = y_c

    if abs(E_r) < 1e-18:
        x_arrow_end = x_arrow_start + 1
        arrow_head = ""
        arrow_label = "E = 0"
    else:
        direction = 1 if E_r > 0 else -1
        x_arrow_end = x_arrow_start + direction * arrow_len

        # Cabeça da seta
        ah = 10
        if direction > 0:
            arrow_head = f"""
                <line x1="{x_arrow_end}" y1="{y_arrow}" x2="{x_arrow_end-ah}" y2="{y_arrow-ah/2}"
                      stroke="{field_color}" stroke-width="3"/>
                <line x1="{x_arrow_end}" y1="{y_arrow}" x2="{x_arrow_end-ah}" y2="{y_arrow+ah/2}"
                      stroke="{field_color}" stroke-width="3"/>
            """
        else:
            arrow_head = f"""
                <line x1="{x_arrow_end}" y1="{y_arrow}" x2="{x_arrow_end+ah}" y2="{y_arrow-ah/2}"
                      stroke="{field_color}" stroke-width="3"/>
                <line x1="{x_arrow_end}" y1="{y_arrow}" x2="{x_arrow_end+ah}" y2="{y_arrow+ah/2}"
                      stroke="{field_color}" stroke-width="3"/>
            """
        arrow_label = f"E = {fmt_num_html(E_r, 'N/C', digits=4)}"

    # Box de cargas
    q_total_html = f"<span style='color:{charge_color(q_total_c)}; font-weight:700;'>Q = {fmt_num_html(q_total_c, 'C')}</span>"
    if is_conductor:
        q_int_html = f"<span style='color:{charge_color(q_int_surface)}; font-weight:700;'>Qint = {fmt_num_html(q_int_surface, 'C')}</span>"
        q_ext_html = f"<span style='color:{charge_color(q_ext_surface)}; font-weight:700;'>Qext = {fmt_num_html(q_ext_surface, 'C')}</span>"
        charge_box_html = f"""
            <div style="font-size:18px; line-height:1.5;">
                {q_total_html}<br>
                {q_int_html}<br>
                {q_ext_html}
            </div>
        """
    else:
        charge_box_html = f"""
            <div style="font-size:18px; line-height:1.5;">
                {q_total_html}
            </div>
        """

    # Box do campo
    field_box_html = f"""
        <div style="font-size:18px; line-height:1.5;">
            <span style='font-weight:700;'>Campo no ponto:</span><br>
            <span style='color:#111827;'>{arrow_label}</span>
        </div>
    """

    # Imagem do logo embutida no HTML, se existir
    logo_data_uri = maybe_base64_image("logo_maua.png")
    logo_html = ""
    if logo_data_uri:
        logo_html = f"""
            <image href="{logo_data_uri}" x="20" y="{svg_h-110}" width="120" height="70" opacity="0.95"/>
        """

    # Elementos para dimensão dos raios
    # Cotas verticais ao lado esquerdo do cilindro
    dim_x_b = 120
    dim_x_a = 165

    def dim_line(x, radius_px, text, color="#111827"):
        y1 = y_c - radius_px
        y2 = y_c + radius_px
        return f"""
            <line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" stroke-width="2"/>
            <line x1="{x-8}" y1="{y1}" x2="{x+8}" y2="{y1}" stroke="{color}" stroke-width="2"/>
            <line x1="{x-8}" y1="{y2}" x2="{x+8}" y2="{y2}" stroke="{color}" stroke-width="2"/>
            <line x1="{x}" y1="{y1}" x2="{x+8}" y2="{y1+10}" stroke="{color}" stroke-width="2"/>
            <line x1="{x}" y1="{y1}" x2="{x-8}" y2="{y1+10}" stroke="{color}" stroke-width="2"/>
            <line x1="{x}" y1="{y2}" x2="{x+8}" y2="{y2-10}" stroke="{color}" stroke-width="2"/>
            <line x1="{x}" y1="{y2}" x2="{x-8}" y2="{y2-10}" stroke="{color}" stroke-width="2"/>
            <text x="{x-18}" y="{y_c}" text-anchor="end" dominant-baseline="middle"
                  font-size="20" font-weight="700" fill="{color}">{text}</text>
        """

    # Linhas laterais do corpo
    outer_rect = f"""
        <rect x="{x_front}" y="{y_c-outer_r_px}" width="{x_back-x_front}" height="{2*outer_r_px}"
              fill="{outer_fill}" stroke="{outer_surface_color}" stroke-width="3" rx="2"/>
    """

    outer_back_ellipse = f"""
        <ellipse cx="{x_back}" cy="{y_c}" rx="{outer_r_px*0.38}" ry="{outer_r_px}"
                 fill="{outer_fill}" stroke="{outer_surface_color}" stroke-width="3"/>
    """

    outer_front_ellipse = f"""
        <ellipse cx="{x_front}" cy="{y_c}" rx="{outer_r_px*0.38}" ry="{outer_r_px}"
                 fill="{outer_fill}" stroke="{outer_surface_color}" stroke-width="4"/>
    """

    # Cavidade interna, se houver
    inner_svg = ""
    if a > 0:
        inner_rect = f"""
            <rect x="{x_front}" y="{y_c-inner_r_px}" width="{x_back-x_front}" height="{2*inner_r_px}"
                  fill="{inner_fill}" stroke="none"/>
        """
        inner_back = f"""
            <ellipse cx="{x_back}" cy="{y_c}" rx="{max(1, inner_r_px*0.38)}" ry="{inner_r_px}"
                     fill="{inner_fill}" stroke="{inner_surface_color}" stroke-width="3"/>
        """
        inner_front = f"""
            <ellipse cx="{x_front}" cy="{y_c}" rx="{max(1, inner_r_px*0.38)}" ry="{inner_r_px}"
                     fill="{inner_fill}" stroke="{inner_surface_color}" stroke-width="4"/>
        """
        # contornos internos do corpo
        inner_edges = f"""
            <line x1="{x_front}" y1="{y_c-inner_r_px}" x2="{x_back}" y2="{y_c-inner_r_px}"
                  stroke="{inner_surface_color}" stroke-width="3"/>
            <line x1="{x_front}" y1="{y_c+inner_r_px}" x2="{x_back}" y2="{y_c+inner_r_px}"
                  stroke="{inner_surface_color}" stroke-width="3"/>
        """
        inner_svg = inner_rect + inner_back + inner_edges + inner_front

    # Superfície gaussiana tracejada
    gauss_svg = ""
    if r > 0:
        gauss_rect = f"""
            <rect x="{x_front}" y="{y_c-gauss_r_px}" width="{x_back-x_front}" height="{2*gauss_r_px}"
                  fill="none" stroke="{gauss_color}" stroke-width="3" stroke-dasharray="10 8"/>
        """
        gauss_back = f"""
            <ellipse cx="{x_back}" cy="{y_c}" rx="{max(1, gauss_r_px*0.38)}" ry="{gauss_r_px}"
                     fill="none" stroke="{gauss_color}" stroke-width="3" stroke-dasharray="10 8"/>
        """
        gauss_front = f"""
            <ellipse cx="{x_front}" cy="{y_c}" rx="{max(1, gauss_r_px*0.38)}" ry="{gauss_r_px}"
                     fill="none" stroke="{gauss_color}" stroke-width="3" stroke-dasharray="10 8"/>
        """
        gauss_svg = gauss_rect + gauss_back + gauss_front

    # Box das cargas dentro do SVG usando foreignObject
    charge_box = f"""
        <foreignObject x="20" y="20" width="280" height="140">
            <div xmlns="http://www.w3.org/1999/xhtml"
                 style="background:white;border:2px solid #d1d5db;border-radius:14px;padding:12px 14px;">
                 <div style="font-weight:800;font-size:18px;margin-bottom:6px;">Cargas</div>
                 {charge_box_html}
            </div>
        </foreignObject>
    """

    # Box do campo
    field_box_x = max(920, int(x_back + gauss_r_px + 40))
    field_box_y = int(y_c - 70)
    field_box = f"""
        <foreignObject x="{field_box_x}" y="{field_box_y}" width="260" height="120">
            <div xmlns="http://www.w3.org/1999/xhtml"
                 style="background:white;border:2px solid #d1d5db;border-radius:14px;padding:12px 14px;">
                 {field_box_html}
            </div>
        </foreignObject>
    """

    # Texto dos parâmetros
    params_text = f"""
        <text x="20" y="{svg_h-22}" font-size="18" fill="#374151">
            a = {fmt_num(a, "m")}   |   b = {fmt_num(b, "m")}   |   r = {fmt_num(r, "m")}
        </text>
    """

    # Seta do campo
    arrow_svg = ""
    if r > 0:
        arrow_svg = f"""
            <line x1="{x_arrow_start}" y1="{y_arrow}" x2="{x_arrow_end}" y2="{y_arrow}"
                  stroke="{field_color}" stroke-width="3"/>
            {arrow_head}
        """

    svg = f"""
    <div class="svg-wrap">
    <svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="0" width="{svg_w}" height="{svg_h}" fill="white"/>

        <!-- Título interno opcional -->
        <text x="{svg_w/2}" y="28" text-anchor="middle" font-size="22" font-weight="800" fill="#111827">
            Cilindro longo e superfície gaussiana
        </text>

        <!-- Dimensões -->
        {dim_line(dim_x_b, outer_r_px, "b")}
        {dim_line(dim_x_a, inner_r_px, "a") if a > 0 else ""}

        <!-- Corpo externo -->
        {outer_back_ellipse}
        {outer_rect}
        {inner_svg}
        {outer_front_ellipse}

        <!-- Superfície gaussiana -->
        {gauss_svg}

        <!-- Campo elétrico -->
        {arrow_svg}

        <!-- Legendas / caixas -->
        {charge_box}
        {field_box}

        <!-- Logo -->
        {logo_html}

        <!-- Parâmetros -->
        {params_text}
    </svg>
    </div>
    """
    return svg

svg_html = build_svg(
    a=a,
    b=b,
    r=r,
    q_total_c=q_total_c,
    is_conductor=is_conductor,
    E_r=E_r,
    qg=qg,
    q_int_surface=q_int_surface,
    q_ext_surface=q_ext_surface,
)

st.markdown(svg_html, unsafe_allow_html=True)

# =========================================================
# Lei de Gauss
# =========================================================
st.markdown('<div class="section-title">Lei de Gauss</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="formula-box">
        <div class="eq-inline"><strong>Φ = ∮ E · dA = q<sub>int</sub> / ε<sub>0</sub></strong></div>
        <div class="small-muted">
            Φ é o fluxo elétrico na superfície gaussiana, E é o campo elétrico,
            A é a área da superfície gaussiana, q<sub>int</sub> é a carga contida
            na superfície gaussiana e ε<sub>0</sub> é a permissividade do vácuo
            = 8,8 × 10<sup>-12</sup> C²/N·m².
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Carga contida na superfície gaussiana
# =========================================================
st.markdown('<div class="section-title">Carga q<sub>gauss</sub> contida na superfície gaussiana</div>', unsafe_allow_html=True)

if r >= b:
    st.markdown(
        f"""
        <div class="formula-box">
            <strong>(i) Superfície gaussiana fora do cilindro</strong><br><br>
            q<sub>gauss</sub> = Q = {fmt_num_html(q_total_c, "C")}
        </div>
        """,
        unsafe_allow_html=True,
    )

elif r < a:
    # Caso adicional necessário fisicamente
    if is_conductor:
        st.markdown(
            f"""
            <div class="formula-box">
                <strong>(ii) Superfície gaussiana na cavidade interna do cilindro condutor</strong><br><br>
                q<sub>gauss</sub> = 0 (sem carga dentro do cilindro)
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="formula-box">
                <strong>Cavidade interna do cilindro</strong><br><br>
                Para r &lt; a, a superfície gaussiana está em uma região sem carga.<br>
                Portanto, q<sub>gauss</sub> = 0.
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    # a <= r < b
    if is_conductor:
        st.markdown(
            f"""
            <div class="formula-box">
                <strong>(iv) Superfície gaussiana no meio da espessura do cilindro condutor</strong><br><br>
                q<sub>gauss</sub> = 0 (toda a carga está na superfície externa do condutor)
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        rho = rho_volume(q_total_c, a, b)
        qg_sub = qg

        st.markdown(
            f"""
            <div class="formula-box">
                <strong>(iii) Superfície gaussiana no meio da espessura do cilindro isolante</strong><br><br>
                ρ = Q / V<sub>total</sub> = q<sub>gauss</sub> / V<sub>r</sub>, sendo V o volume e ρ a densidade de carga volumétrica.<br><br>

                Q / [π (b² − a²) L] = q<sub>gauss</sub> / [π (r² − a²) L]<br><br>

                q<sub>gauss</sub> = Q (r² − a²) / (b² − a²)<br><br>

                q<sub>gauss</sub> = ({fmt_num_html(q_total_c, "C")}) · ({fmt_num_html(r**2, "m²")} − {fmt_num_html(a**2, "m²")})
                / ({fmt_num_html(b**2, "m²")} − {fmt_num_html(a**2, "m²")})<br><br>

                <strong>q<sub>gauss</sub> = {fmt_num_html(qg_sub, "C")}</strong><br><br>

                ρ = {fmt_num_html(rho, "C/m³")}
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# Área da superfície gaussiana
# =========================================================
st.markdown('<div class="section-title">Área da superfície gaussiana</div>', unsafe_allow_html=True)

if r > 0:
    st.markdown(
        f"""
        <div class="formula-box">
            A = 2πrL<br><br>
            A = 2π · {fmt_num_html(r, "m")} · {fmt_num_html(L_REF, "m")}<br><br>
            <strong>A = {fmt_num_html(A_g, "m²")}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="formula-box">
            Para r = 0, a área da superfície gaussiana cilíndrica não é definida.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# Campo elétrico
# =========================================================
st.markdown('<div class="section-title">Campo elétrico</div>', unsafe_allow_html=True)

if r > 0:
    st.markdown(
        f"""
        <div class="formula-box">
            <strong>Lei de Gauss no caso de simetria</strong>: campo constante E em toda a superfície gaussiana
            e sempre paralelo ao vetor área.<br><br>

            EA = q<sub>gauss</sub> / ε<sub>0</sub><br><br>

            E = q<sub>gauss</sub> / (A ε<sub>0</sub>)<br><br>

            E = {fmt_num_html(qg, "C")} / ({fmt_num_html(A_g, "m²")} · 8,8 × 10<sup>-12</sup> C²/N·m²)<br><br>

            <strong>E = {fmt_num_html(E_r, "N/C")}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="formula-box">
            Para r = 0, o campo no eixo é tomado como 0 por simetria no gráfico e na análise numérica do app.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# Gráfico E x r
# =========================================================
st.markdown('<div class="section-title">Gráfico</div>', unsafe_allow_html=True)

r_max_graph = max(b + 2.0, 1.8 * b + 1.0, 2.0)
rr = np.linspace(0.001, r_max_graph, 700)
EE = np.array([electric_field(float(rv), a, b, q_total_c, is_conductor) for rv in rr])

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=rr,
        y=EE,
        mode="lines",
        name="E(r)",
        line=dict(width=3),
        hovertemplate="r = %{x:.3f} m<br>E = %{y:.6g} N/C<extra></extra>",
    )
)

# marca a posição atual do slider
fig.add_trace(
    go.Scatter(
        x=[r],
        y=[E_r],
        mode="markers",
        name="Ponto selecionado",
        marker=dict(size=11),
        hovertemplate="r = %{x:.3f} m<br>E = %{y:.6g} N/C<extra></extra>",
    )
)

# Linhas de referência em a e b
fig.add_vline(x=a, line_width=1.5, line_dash="dash", line_color="#6b7280")
fig.add_vline(x=b, line_width=1.5, line_dash="dash", line_color="#111827")

fig.update_layout(
    xaxis_title="Distância radial r (m)",
    yaxis_title="Campo elétrico E (N/C)",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    height=430,
)

fig.update_xaxes(
    showgrid=True,
    gridcolor="#e5e7eb",
    zeroline=True,
    zerolinecolor="#9ca3af",
)

fig.update_yaxes(
    showgrid=True,
    gridcolor="#e5e7eb",
    zeroline=True,
    zerolinecolor="#9ca3af",
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Rodapé
# =========================================================
st.markdown(
    """
    <div class="small-muted" style="margin-top: 0.8rem;">
        Observações:
        <ul>
            <li>O simulador adota <strong>L = 1 m</strong> como comprimento de referência para o cilindro longo.</li>
            <li>Para <strong>condutor</strong>, a carga está toda na superfície externa quando não há carga na cavidade interna.</li>
            <li>O gráfico mostra o <strong>campo elétrico assinado</strong>: positivo para fora e negativo para dentro.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
