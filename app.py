import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# CONFIGURAÇÃO
# =========================================================
st.set_page_config(
    page_title="Simulador Campo Elétrico Cilindro",
    page_icon="⚡",
    layout="wide",
)

EPSILON_0 = 8.8e-12  # conforme solicitado


# =========================================================
# FORMATAÇÃO
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


def exp_to_superscript(n: int) -> str:
    return "".join(SUPERSCRIPTS.get(ch, ch) for ch in str(n))


def fmt_num(x: float, digits: int = 4) -> str:
    """Formata número sem notação e/E."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "indefinido"

    if abs(x) < 1e-15:
        return "0"

    ax = abs(x)

    if 1e-3 <= ax < 1e4:
        s = f"{x:.{digits}f}".rstrip("0").rstrip(".")
        return s

    exp = int(math.floor(math.log10(ax)))
    mant = x / (10 ** exp)
    s_mant = f"{mant:.{digits}f}".rstrip("0").rstrip(".")
    return f"{s_mant} × 10{exp_to_superscript(exp)}"


def fmt_num_unit(x: float, unit: str = "", digits: int = 4) -> str:
    base = fmt_num(x, digits)
    return f"{base} {unit}".strip()


def fmt_html(x: float, unit: str = "", digits: int = 4) -> str:
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


def charge_color(q_coulomb: float) -> str:
    if q_coulomb > 0:
        return "#d62828"   # vermelho
    elif q_coulomb < 0:
        return "#1d4ed8"   # azul
    return "#111111"       # preto


def soft_fill(hex_color: str, alpha: float = 0.18) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(0,0,0,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# =========================================================
# FÍSICA
# =========================================================
def lambda_linear(q_total_c: float, L: float) -> float:
    if L <= 0:
        return 0.0
    return q_total_c / L


def q_gauss(r: float, a: float, b: float, q_total_c: float, is_conductor: bool) -> float:
    """
    Q é a carga total no trecho de comprimento L selecionado pelo usuário.
    Dentro do material isolante:
      q_gauss = Q * (r² - a²)/(b² - a²)
    """
    if r < 0:
        return 0.0

    if r < a:
        return 0.0

    if a <= r < b:
        if is_conductor:
            return 0.0
        denom = (b**2 - a**2)
        if abs(denom) < 1e-15:
            return 0.0
        return q_total_c * (r**2 - a**2) / denom

    return q_total_c


def area_gauss(r: float, L: float) -> float:
    return 2 * math.pi * r * L


def electric_field(r: float, a: float, b: float, q_total_c: float, L: float, is_conductor: bool) -> float:
    """
    Campo assinado:
      positivo -> radial para fora
      negativo -> radial para dentro
    """
    if r <= 0 or L <= 0:
        return 0.0

    qg = q_gauss(r, a, b, q_total_c, is_conductor)
    A = area_gauss(r, L)
    return qg / (EPSILON_0 * A)


def rho_volume(q_total_c: float, a: float, b: float, L: float) -> float:
    V = math.pi * (b**2 - a**2) * L
    if abs(V) < 1e-15:
        return 0.0
    return q_total_c / V


# =========================================================
# ESTILO
# =========================================================
st.markdown(
    """
    <style>
        .stApp {
            background: #f5f7fb;
            color: #111827;
        }

        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }

        .main-title {
            font-size: 2rem;
            font-weight: 800;
            color: #111827;
            line-height: 1.15;
            margin-bottom: 0.15rem;
        }

        .main-subtitle {
            font-size: 1.05rem;
            color: #374151;
            margin-top: 0.1rem;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: #111827;
            margin-top: 1.25rem;
            margin-bottom: 0.55rem;
        }

        .white-card {
            background: #ffffff;
            border: 1px solid #d1d5db;
            border-radius: 16px;
            padding: 1rem 1rem;
            color: #111827 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.04);
            margin-bottom: 0.8rem;
        }

        .formula-card {
            background: #ffffff;
            border: 1px solid #d1d5db;
            border-left: 5px solid #2563eb;
            border-radius: 14px;
            padding: 0.95rem 1rem;
            color: #111827 !important;
            margin-bottom: 0.8rem;
            overflow-wrap: anywhere;
        }

        .slider-card {
            background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%);
            border: 2px solid #60a5fa;
            border-radius: 18px;
            padding: 1rem 1rem 0.5rem 1rem;
            margin-top: 0.5rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.08);
        }

        .small-note {
            color: #4b5563;
            font-size: 0.95rem;
        }

        .equation {
            font-size: 1.03rem;
            line-height: 1.8;
            color: #111827 !important;
        }

        .black-text * {
            color: #111827 !important;
        }

        /* sliders */
        .stSlider > div > div > div {
            color: #111827 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HEADER
# =========================================================
col_logo, col_title = st.columns([1, 4], vertical_alignment="center")

with col_logo:
    logo_path = Path("logo_maua.png")
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.markdown(
            """
            <div class="white-card" style="text-align:center;">
                <strong>logo_maua.png</strong><br>não encontrado
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_title:
    st.markdown('<div class="main-title">Simulador Campo Elétrico Cilindro</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Estude o campo elétrico de um cilindro longo.</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="white-card black-text">
        <div class="small-note">
            Neste app, <strong>Q</strong> é a carga total contida no trecho cilíndrico de comprimento
            <strong>L</strong> escolhido no simulador. Assim, no cálculo do campo elétrico,
            aparece naturalmente a razão <strong>Q/L</strong>, isto é, a carga por unidade de comprimento.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# ESTADO DOS SLIDERS
# =========================================================
if "a" not in st.session_state:
    st.session_state.a = 0.5

a = st.session_state.a

b_min = round(a + 0.5, 2)
if "b" not in st.session_state:
    st.session_state.b = max(1.0, b_min)
if st.session_state.b < b_min:
    st.session_state.b = b_min

if "Q_micro" not in st.session_state:
    st.session_state.Q_micro = 6.0

if "L" not in st.session_state:
    st.session_state.L = 1.0

# r depende de b, então garantimos coerência
r_max_pre = max(2.0, st.session_state.b + 2.0, 1.6 * st.session_state.b + 1.0)
if "r" not in st.session_state:
    st.session_state.r = min(max(0.8 * st.session_state.b, 0.05), r_max_pre)

# =========================================================
# PARÂMETROS
# =========================================================
st.markdown('<div class="section-title">Parâmetros</div>', unsafe_allow_html=True)

p1, p2 = st.columns(2)

with p1:
    st.markdown('<div class="white-card black-text">', unsafe_allow_html=True)

    a = st.slider(
        "Raio interno a do cilindro (m)",
        min_value=0.0,
        max_value=5.0,
        value=float(st.session_state.a),
        step=0.05,
        key="a",
        help="Pode ser zero para cilindro maciço."
    )

    b_min = round(a + 0.5, 2)
    b_max = max(6.0, round(a + 5.0, 2))
    if st.session_state.b < b_min:
        st.session_state.b = b_min

    b = st.slider(
        "Raio externo b do cilindro (m)",
        min_value=float(b_min),
        max_value=float(b_max),
        value=float(st.session_state.b),
        step=0.05,
        key="b",
        help="O app garante sempre b ≥ a + 0,5 m."
    )

    Q_micro = st.slider(
        "Carga Q do cilindro (micronC)",
        min_value=-20.0,
        max_value=20.0,
        value=float(st.session_state.Q_micro),
        step=0.1,
        key="Q_micro",
    )

    L = st.slider(
        "Comprimento L do trecho do cilindro (m)",
        min_value=0.2,
        max_value=10.0,
        value=float(st.session_state.L),
        step=0.1,
        key="L",
        help="Adicionado para evitar assumir L = 1 m e permitir valor numérico coerente do campo."
    )

    is_conductor = st.toggle("Considerar o cilindro como condutor", value=False)
    st.markdown('</div>', unsafe_allow_html=True)

with p2:
    r_max = max(2.0, b + 2.0, 1.6 * b + 1.0)
    if st.session_state.r > r_max:
        st.session_state.r = r_max

    st.markdown('<div class="slider-card black-text">', unsafe_allow_html=True)
    r = st.slider(
        "Raio da superfície gaussiana r (m) para estudo do campo elétrico",
        min_value=0.0,
        max_value=float(round(r_max, 2)),
        value=float(round(st.session_state.r, 2)),
        step=0.01,
        key="r",
    )
    st.markdown(
        f"""
        <div class="small-note">
            Faixa automática do slider: 0 até {fmt_num_unit(r_max, "m")}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

q_total_c = Q_micro * 1e-6
lam = lambda_linear(q_total_c, L)

# =========================================================
# CÁLCULOS
# =========================================================
qg = q_gauss(r, a, b, q_total_c, is_conductor)
A_g = area_gauss(r, L) if r > 0 else 0.0
E_r = electric_field(r, a, b, q_total_c, L, is_conductor)
rho = rho_volume(q_total_c, a, b, L) if not is_conductor else 0.0

if is_conductor:
    q_int_surface = 0.0
    q_ext_surface = q_total_c
else:
    q_int_surface = None
    q_ext_surface = None


# =========================================================
# IMAGEM SVG - renderizada corretamente com components.html
# =========================================================
st.markdown('<div class="section-title">Imagem</div>', unsafe_allow_html=True)

def build_svg(a, b, r, q_total_c, E_r, qg, is_conductor, q_int_surface, q_ext_surface):
    px_per_m = 85.0  # escala fixa

    outer_r_px = max(8.0, b * px_per_m)
    inner_r_px = a * px_per_m
    gauss_r_px = r * px_per_m

    x_front = 260
    x_back = 820
    y_c = 260

    svg_w = int(max(1500, x_back + max(outer_r_px, gauss_r_px) + 360))
    svg_h = int(max(620, y_c + max(outer_r_px, gauss_r_px) + 160))

    q_color = charge_color(q_total_c)
    ext_color = q_color
    int_color = charge_color(0.0) if is_conductor else q_color

    ext_fill = soft_fill(ext_color, 0.18)
    gauss_color = "#16a34a"
    field_color = "#111827"
    gray = "#111827"

    # seta do campo
    x_point = x_back + gauss_r_px
    y_point = y_c
    arrow_len = 85

    if abs(E_r) < 1e-18:
        x_arrow_end = x_point + 1
        arrow_svg = ""
        field_text = f"E = 0 N/C"
    else:
        direction = 1 if E_r > 0 else -1
        x_arrow_end = x_point + direction * arrow_len
        ah = 10
        if direction > 0:
            head = f"""
            <line x1="{x_arrow_end}" y1="{y_point}" x2="{x_arrow_end-ah}" y2="{y_point-ah/2}" stroke="{field_color}" stroke-width="3"/>
            <line x1="{x_arrow_end}" y1="{y_point}" x2="{x_arrow_end-ah}" y2="{y_point+ah/2}" stroke="{field_color}" stroke-width="3"/>
            """
        else:
            head = f"""
            <line x1="{x_arrow_end}" y1="{y_point}" x2="{x_arrow_end+ah}" y2="{y_point-ah/2}" stroke="{field_color}" stroke-width="3"/>
            <line x1="{x_arrow_end}" y1="{y_point}" x2="{x_arrow_end+ah}" y2="{y_point+ah/2}" stroke="{field_color}" stroke-width="3"/>
            """
        arrow_svg = f"""
        <line x1="{x_point}" y1="{y_point}" x2="{x_arrow_end}" y2="{y_point}" stroke="{field_color}" stroke-width="3"/>
        {head}
        """
        field_text = f"E = {fmt_num(E_r)} N/C"

    # boxes
    charge_box_x = 20
    charge_box_y = 20
    charge_box_w = 340
    charge_box_h = 125 if not is_conductor else 165

    field_box_x = int(max(1100, x_back + gauss_r_px + 40))
    field_box_y = int(y_c - 70)
    field_box_w = 290
    field_box_h = 110

    # linhas de cota
    dim_x_b = 135
    dim_x_a = 185

    def dimension(x, radius, label):
        y1 = y_c - radius
        y2 = y_c + radius
        return f"""
        <line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{gray}" stroke-width="2"/>
        <line x1="{x-8}" y1="{y1}" x2="{x+8}" y2="{y1}" stroke="{gray}" stroke-width="2"/>
        <line x1="{x-8}" y1="{y2}" x2="{x+8}" y2="{y2}" stroke="{gray}" stroke-width="2"/>
        <line x1="{x}" y1="{y1}" x2="{x+8}" y2="{y1+10}" stroke="{gray}" stroke-width="2"/>
        <line x1="{x}" y1="{y1}" x2="{x-8}" y2="{y1+10}" stroke="{gray}" stroke-width="2"/>
        <line x1="{x}" y1="{y2}" x2="{x+8}" y2="{y2-10}" stroke="{gray}" stroke-width="2"/>
        <line x1="{x}" y1="{y2}" x2="{x-8}" y2="{y2-10}" stroke="{gray}" stroke-width="2"/>
        <text x="{x-18}" y="{y_c}" text-anchor="end" dominant-baseline="middle"
              font-size="22" font-weight="700" fill="{gray}">{label}</text>
        """

    # corpo externo
    outer_back = f"""
    <ellipse cx="{x_back}" cy="{y_c}" rx="{outer_r_px*0.38}" ry="{outer_r_px}"
             fill="{ext_fill}" stroke="{ext_color}" stroke-width="3"/>
    """
    outer_rect = f"""
    <rect x="{x_front}" y="{y_c-outer_r_px}" width="{x_back-x_front}" height="{2*outer_r_px}"
          fill="{ext_fill}" stroke="{ext_color}" stroke-width="3"/>
    """
    outer_front = f"""
    <ellipse cx="{x_front}" cy="{y_c}" rx="{outer_r_px*0.38}" ry="{outer_r_px}"
             fill="{ext_fill}" stroke="{ext_color}" stroke-width="4"/>
    """

    inner_svg = ""
    if a > 0:
        inner_svg = f"""
        <rect x="{x_front}" y="{y_c-inner_r_px}" width="{x_back-x_front}" height="{2*inner_r_px}"
              fill="#ffffff" stroke="none"/>
        <ellipse cx="{x_back}" cy="{y_c}" rx="{max(1, inner_r_px*0.38)}" ry="{inner_r_px}"
                 fill="#ffffff" stroke="{int_color}" stroke-width="3"/>
        <line x1="{x_front}" y1="{y_c-inner_r_px}" x2="{x_back}" y2="{y_c-inner_r_px}"
              stroke="{int_color}" stroke-width="3"/>
        <line x1="{x_front}" y1="{y_c+inner_r_px}" x2="{x_back}" y2="{y_c+inner_r_px}"
              stroke="{int_color}" stroke-width="3"/>
        <ellipse cx="{x_front}" cy="{y_c}" rx="{max(1, inner_r_px*0.38)}" ry="{inner_r_px}"
                 fill="#ffffff" stroke="{int_color}" stroke-width="4"/>
        """

    # superfície gaussiana
    gauss_svg = ""
    if r > 0:
        gauss_svg = f"""
        <rect x="{x_front}" y="{y_c-gauss_r_px}" width="{x_back-x_front}" height="{2*gauss_r_px}"
              fill="none" stroke="{gauss_color}" stroke-width="3" stroke-dasharray="10 8"/>
        <ellipse cx="{x_back}" cy="{y_c}" rx="{max(1, gauss_r_px*0.38)}" ry="{gauss_r_px}"
                 fill="none" stroke="{gauss_color}" stroke-width="3" stroke-dasharray="10 8"/>
        <ellipse cx="{x_front}" cy="{y_c}" rx="{max(1, gauss_r_px*0.38)}" ry="{gauss_r_px}"
                 fill="none" stroke="{gauss_color}" stroke-width="3" stroke-dasharray="10 8"/>
        """

    # texto do box de cargas
    if is_conductor:
        charges_lines = [
            ("Q =", fmt_num_unit(q_total_c, "C"), charge_color(q_total_c)),
            ("Qint =", fmt_num_unit(q_int_surface, "C"), charge_color(q_int_surface)),
            ("Qext =", fmt_num_unit(q_ext_surface, "C"), charge_color(q_ext_surface)),
        ]
    else:
        charges_lines = [
            ("Q =", fmt_num_unit(q_total_c, "C"), charge_color(q_total_c)),
        ]

    charge_text_svg = ""
    base_y = charge_box_y + 36
    charge_text_svg += f'<text x="{charge_box_x+14}" y="{base_y}" font-size="18" font-weight="800" fill="#111827">Cargas</text>'
    for i, (lab, val, col) in enumerate(charges_lines):
        yy = base_y + 28 + i * 26
        charge_text_svg += f'<text x="{charge_box_x+14}" y="{yy}" font-size="18" fill="{col}" font-weight="700">{lab} {val}</text>'

    # box do campo
    field_box_svg = f"""
    <rect x="{field_box_x}" y="{field_box_y}" width="{field_box_w}" height="{field_box_h}"
          rx="14" ry="14" fill="#ffffff" stroke="#d1d5db" stroke-width="2"/>
    <text x="{field_box_x+14}" y="{field_box_y+30}" font-size="18" font-weight="800" fill="#111827">Campo no ponto</text>
    <text x="{field_box_x+14}" y="{field_box_y+62}" font-size="18" fill="#111827">{field_text}</text>
    """

    # box de cargas
    charge_box_svg = f"""
    <rect x="{charge_box_x}" y="{charge_box_y}" width="{charge_box_w}" height="{charge_box_h}"
          rx="14" ry="14" fill="#ffffff" stroke="#d1d5db" stroke-width="2"/>
    {charge_text_svg}
    """

    # parâmetros embaixo
    params_text = f"""
    <text x="20" y="{svg_h-24}" font-size="18" fill="#111827" font-weight="600">
        a = {fmt_num(a)} m   |   b = {fmt_num(b)} m   |   r = {fmt_num(r)} m   |   L = {fmt_num(L)} m
    </text>
    """

    svg = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8"/>
    <style>
        body {{
            margin:0;
            background:#ffffff;
            font-family: Arial, Helvetica, sans-serif;
        }}
        .wrap {{
            width:100%;
            overflow-x:auto;
            overflow-y:hidden;
            background:#ffffff;
            border:1px solid #d1d5db;
            border-radius:16px;
        }}
        svg {{
            display:block;
            background:#ffffff;
        }}
    </style>
    </head>
    <body>
      <div class="wrap">
        <svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg">
            <rect x="0" y="0" width="{svg_w}" height="{svg_h}" fill="#ffffff"/>
            <text x="{svg_w/2}" y="30" text-anchor="middle" font-size="24" font-weight="800" fill="#111827">
                Cilindro longo e superfície gaussiana
            </text>

            {dimension(dim_x_b, outer_r_px, "b")}
            {dimension(dim_x_a, inner_r_px, "a") if a > 0 else ""}

            {outer_back}
            {outer_rect}
            {inner_svg}
            {outer_front}

            {gauss_svg}
            {arrow_svg}

            {charge_box_svg}
            {field_box_svg}

            {params_text}
        </svg>
      </div>
    </body>
    </html>
    """
    return svg, svg_h + 20


svg_html, svg_height = build_svg(
    a=a,
    b=b,
    r=r,
    q_total_c=q_total_c,
    E_r=E_r,
    qg=qg,
    is_conductor=is_conductor,
    q_int_surface=q_int_surface,
    q_ext_surface=q_ext_surface,
)

# IMPORTANTE: components.html evita o problema do HTML/SVG aparecer como texto
components.html(svg_html, height=svg_height, scrolling=True)

# =========================================================
# LEI DE GAUSS
# =========================================================
st.markdown('<div class="section-title">Lei de Gauss</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="formula-card black-text">
        <div class="equation">
            <strong>Φ = ∮ E · dA = q<sub>int</sub> / ε<sub>0</sub></strong><br>
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
# CARGA q_gauss
# =========================================================
st.markdown('<div class="section-title">Carga q<sub>gauss</sub> contida na superfície gaussiana</div>', unsafe_allow_html=True)

if r >= b:
    st.markdown(
        f"""
        <div class="formula-card black-text">
            <div class="equation">
                <strong>(i) Se a superfície gaussiana estiver fora do cilindro</strong><br><br>
                q<sub>gauss</sub> = Q = {fmt_html(q_total_c, "C")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif r < a:
    if is_conductor:
        st.markdown(
            """
            <div class="formula-card black-text">
                <div class="equation">
                    <strong>(ii) Se a superfície gaussiana estiver dentro do cilindro condutor</strong><br><br>
                    q<sub>gauss</sub> = 0 (sem carga dentro do cilindro)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="formula-card black-text">
                <div class="equation">
                    Para r &lt; a, a superfície gaussiana está na cavidade interna.<br>
                    Portanto, q<sub>gauss</sub> = 0.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    if is_conductor:
        st.markdown(
            """
            <div class="formula-card black-text">
                <div class="equation">
                    <strong>(iv) Se a superfície gaussiana estiver no meio da espessura do cilindro condutor</strong><br><br>
                    q<sub>gauss</sub> = 0 (toda a carga está na superfície externa do condutor)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="formula-card black-text">
                <div class="equation">
                    <strong>(iii) Se a superfície gaussiana estiver no meio da espessura do cilindro isolante</strong><br><br>

                    ρ = Q / V<sub>total</sub> = q<sub>gauss</sub> / V<sub>r</sub>, sendo V o volume e ρ a densidade de carga volumétrica.<br><br>

                    Q / [π (b² − a²) L] = q<sub>gauss</sub> / [π (r² − a²) L]<br><br>

                    q<sub>gauss</sub> = Q (r² − a²) / (b² − a²)<br><br>

                    q<sub>gauss</sub> = ({fmt_html(q_total_c, "C")}) · ({fmt_html(r**2, "m²")} − {fmt_html(a**2, "m²")})
                    / ({fmt_html(b**2, "m²")} − {fmt_html(a**2, "m²")})<br><br>

                    <strong>q<sub>gauss</sub> = {fmt_html(qg, "C")}</strong><br><br>

                    ρ = {fmt_html(rho, "C/m³")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# ÁREA
# =========================================================
st.markdown('<div class="section-title">Área da superfície gaussiana</div>', unsafe_allow_html=True)

if r > 0:
    st.markdown(
        f"""
        <div class="formula-card black-text">
            <div class="equation">
                A = 2πrL<br><br>
                A = 2π · {fmt_html(r, "m")} · {fmt_html(L, "m")}<br><br>
                <strong>A = {fmt_html(A_g, "m²")}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="formula-card black-text">
            <div class="equation">
                Para r = 0, a área da superfície gaussiana cilíndrica não é definida.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# CAMPO ELÉTRICO
# =========================================================
st.markdown('<div class="section-title">Campo elétrico</div>', unsafe_allow_html=True)

if r > 0:
    if r >= b:
        field_symbolic = f"""
        E A = q<sub>gauss</sub> / ε<sub>0</sub><br><br>
        E = Q / (2πrLε<sub>0</sub>)<br><br>
        E = (Q/L) / (2πrε<sub>0</sub>)<br><br>
        E = ({fmt_html(q_total_c / L, "C/m")}) / (2π · {fmt_html(r, "m")} · 8,8 × 10<sup>-12</sup> C²/N·m²)<br><br>
        <strong>E = {fmt_html(E_r, "N/C")}</strong>
        """
    elif r < a:
        field_symbolic = """
        E A = q<sub>gauss</sub> / ε<sub>0</sub><br><br>
        Como q<sub>gauss</sub> = 0, então:<br><br>
        <strong>E = 0 N/C</strong>
        """
    else:
        if is_conductor:
            field_symbolic = """
            E A = q<sub>gauss</sub> / ε<sub>0</sub><br><br>
            Como q<sub>gauss</sub> = 0, então:<br><br>
            <strong>E = 0 N/C</strong>
            """
        else:
            field_symbolic = f"""
            Lei de Gauss no caso de simetria: campo constante E em toda superfície gaussiana
            e sempre paralelo ao vetor área.<br><br>

            E A = q<sub>gauss</sub> / ε<sub>0</sub><br><br>

            E = q<sub>gauss</sub> / (A ε<sub>0</sub>)<br><br>

            E = [Q (r² − a²)/(b² − a²)] / (2πrLε<sub>0</sub>)<br><br>

            E = [(Q/L) (r² − a²)] / [2πr (b² − a²) ε<sub>0</sub>]<br><br>

            E = [({fmt_html(q_total_c / L, "C/m")}) · ({fmt_html(r**2, "m²")} − {fmt_html(a**2, "m²")})]
            / [2π · {fmt_html(r, "m")} · ({fmt_html(b**2, "m²")} − {fmt_html(a**2, "m²")}) · 8,8 × 10<sup>-12</sup> C²/N·m²]<br><br>

            <strong>E = {fmt_html(E_r, "N/C")}</strong>
            """

    st.markdown(
        f"""
        <div class="formula-card black-text">
            <div class="equation">
                {field_symbolic}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="formula-card black-text">
            <div class="equation">
                Para r = 0, o app exibe o campo como 0 por convenção visual no eixo.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# GRÁFICO
# =========================================================
st.markdown('<div class="section-title">Gráfico</div>', unsafe_allow_html=True)

r_max_graph = max(b + 2.0, 1.8 * b + 1.0, 2.0)
rr = np.linspace(0.001, r_max_graph, 900)
EE = np.array([electric_field(float(rv), a, b, q_total_c, L, is_conductor) for rv in rr])

# faixas automáticas calculadas no código
y_abs = np.nanmax(np.abs(EE)) if len(EE) > 0 else 1.0
if y_abs < 1e-9:
    y_abs = 1.0
y_margin = 0.12 * y_abs

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=rr,
        y=EE,
        mode="lines",
        name="E(r)",
        line=dict(width=3, color="#2563eb"),
        hovertemplate="r = %{x:.3f} m<br>E = %{y:.6g} N/C<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=[r],
        y=[E_r],
        mode="markers",
        name="Ponto selecionado",
        marker=dict(size=11, color="#d62828"),
        hovertemplate="r = %{x:.3f} m<br>E = %{y:.6g} N/C<extra></extra>",
    )
)

fig.add_vline(x=a, line_width=1.5, line_dash="dash", line_color="#111111")
fig.add_vline(x=b, line_width=1.5, line_dash="dash", line_color="#444444")

fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=20, r=20, t=20, b=20),
    height=440,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        font=dict(color="black")
    ),
    dragmode=False,
)

fig.update_xaxes(
    title="Distância radial r (m)",
    title_font=dict(color="black"),
    tickfont=dict(color="black"),
    linecolor="black",
    mirror=True,
    showgrid=True,
    gridcolor="#e5e7eb",
    zeroline=True,
    zerolinecolor="#9ca3af",
    range=[0, r_max_graph],
    fixedrange=True,
)

fig.update_yaxes(
    title="Campo elétrico E (N/C)",
    title_font=dict(color="black"),
    tickfont=dict(color="black"),
    linecolor="black",
    mirror=True,
    showgrid=True,
    gridcolor="#e5e7eb",
    zeroline=True,
    zerolinecolor="#9ca3af",
    range=[-y_abs - y_margin, y_abs + y_margin],
    fixedrange=True,
)

graph_config = {
    "displaylogo": False,
    "scrollZoom": False,
    "doubleClick": False,
    "modeBarButtonsToRemove": [
        "zoom2d",
        "pan2d",
        "select2d",
        "lasso2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d",
    ],
    "responsive": True,
}

st.plotly_chart(fig, use_container_width=True, config=graph_config)

# =========================================================
# OBSERVAÇÃO FINAL
# =========================================================
st.markdown(
    """
    <div class="white-card black-text">
        <div class="small-note">
            <strong>Correções implementadas nesta versão:</strong>
            <ul>
                <li>Renderização da imagem com <code>components.html(...)</code>, evitando que o SVG apareça como texto bruto.</li>
                <li>Remoção de <code>foreignObject</code>, que costuma causar incompatibilidades.</li>
                <li>Cards e fórmulas com fundo branco e texto preto, evitando sumiço de texto em tema escuro.</li>
                <li>Parâmetros em sliders.</li>
                <li>Inclusão de <strong>L</strong> como slider para evitar assumir <strong>L = 1</strong>.</li>
                <li>Gráfico com eixos automáticos definidos por código, porém sem permitir zoom manual.</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
