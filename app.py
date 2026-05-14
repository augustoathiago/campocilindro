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
    """Formata sem e/E."""
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
    return f"{fmt_num(x, digits)} {unit}".strip()


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


def charge_color(value: float) -> str:
    if value > 0:
        return "#d62828"  # vermelho
    elif value < 0:
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


# =========================================================
# FÍSICA
# =========================================================
def lambda_equiv(rho_c_m3: float, a: float, b: float) -> float:
    """
    Carga por unidade de comprimento equivalente:
    λ = ρ π (b² - a²)

    Para condutor:
    interpretamos ρ como densidade volumétrica equivalente para definir
    a carga total por unidade de comprimento λ que, fisicamente, ficará
    toda na superfície externa.
    """
    return rho_c_m3 * math.pi * (b**2 - a**2)


def q_gauss_coeff_per_L(r: float, a: float, b: float, rho_c_m3: float, is_conductor: bool) -> float:
    """
    Retorna o coeficiente de q_gauss/L em C/m.
    Assim:
      q_gauss = [coeficiente] * L

    Casos:
      - r < a            -> 0
      - a <= r < b:
          * condutor     -> 0
          * isolante     -> ρ π (r² - a²)
      - r >= b           -> ρ π (b² - a²)
    """
    if r < 0:
        return 0.0

    if r < a:
        return 0.0

    if a <= r < b:
        if is_conductor:
            return 0.0
        return rho_c_m3 * math.pi * (r**2 - a**2)

    return rho_c_m3 * math.pi * (b**2 - a**2)


def electric_field(r: float, a: float, b: float, rho_c_m3: float, is_conductor: bool) -> float:
    """
    Campo elétrico assinado:
      positivo -> radial para fora
      negativo -> radial para dentro

    Como:
      q_gauss = coef * L
      A = 2πrL
    então o L cancela.
    """
    if r <= 0:
        return 0.0

    q_coeff = q_gauss_coeff_per_L(r, a, b, rho_c_m3, is_conductor)  # C/m
    return q_coeff / (2 * math.pi * r * EPSILON_0)


def rho_to_total_Q_symbolic(rho_c_m3: float, a: float, b: float) -> float:
    """
    Apenas para exibir λ = Q/L = ρ π (b²-a²)
    """
    return lambda_equiv(rho_c_m3, a, b)


# =========================================================
# CSS
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

        /* Corrige textos claros sumindo em parâmetros */
        label, p, div, span {
            color: inherit;
        }

        [data-testid="stWidgetLabel"] p,
        [data-testid="stMarkdownContainer"] p,
        .stSlider label,
        .stRadio label,
        .stToggle label,
        .stSelectbox label,
        .stNumberInput label {
            color: #111827 !important;
        }

        .stSlider > div > div,
        .stRadio > div,
        .stToggle > div {
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
            Nesta versão, o simulador usa a <strong>densidade volumétrica de carga ρ</strong>.
            Assim, o cálculo do campo elétrico fica independente do comprimento
            porque o fator <strong>L</strong> cancela entre a carga contida e a área da superfície gaussiana.
            <br><br>
            <strong>No caso condutor</strong>, ρ é tratado como uma <strong>densidade volumétrica equivalente</strong>
            apenas para definir a carga por unidade de comprimento
            <strong>λ = ρπ(b² − a²)</strong>, que fisicamente fica toda na superfície externa.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# ESTADO
# =========================================================
if "a" not in st.session_state:
    st.session_state.a = 0.4

if "b" not in st.session_state:
    st.session_state.b = 1.0

if "rho_micro" not in st.session_state:
    st.session_state.rho_micro = 6.0

if "r" not in st.session_state:
    st.session_state.r = 0.8

a = float(st.session_state.a)
b_min = round(a + 0.5, 2)
b_max = 2.0
if st.session_state.b < b_min:
    st.session_state.b = b_min
if st.session_state.b > b_max:
    st.session_state.b = b_max

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
        max_value=1.5,
        value=float(st.session_state.a),
        step=0.05,
        key="a",
        help="Pode ser zero para cilindro maciço."
    )

    b_min = round(a + 0.5, 2)
    b_max = 2.0

    if st.session_state.b < b_min:
        st.session_state.b = b_min
    if st.session_state.b > b_max:
        st.session_state.b = b_max

    b = st.slider(
        "Raio externo b do cilindro (m)",
        min_value=float(b_min),
        max_value=float(b_max),
        value=float(st.session_state.b),
        step=0.05,
        key="b",
        help="O app garante sempre b ≥ a + 0,5 m e b ≤ 2,0 m."
    )

    rho_micro = st.slider(
        "Densidade volumétrica de carga ρ (microC/m³)",
        min_value=-20.0,
        max_value=20.0,
        value=float(st.session_state.rho_micro),
        step=0.1,
        key="rho_micro",
    )

    material = st.radio(
        "Material do cilindro",
        options=["Isolante", "Condutor"],
        horizontal=False,
    )

    is_conductor = material == "Condutor"

    st.markdown('</div>', unsafe_allow_html=True)

with p2:
    r_max = 2.5
    if st.session_state.r > r_max:
        st.session_state.r = r_max

    st.markdown('<div class="slider-card black-text">', unsafe_allow_html=True)
    r = st.slider(
        "Raio da superfície gaussiana r (m) para estudo do campo elétrico",
        min_value=0.0,
        max_value=float(r_max),
        value=float(round(st.session_state.r, 2)),
        step=0.01,
        key="r",
    )
    st.markdown(
        """
        <div class="small-note">
            Faixa limitada de 0 até 2,5 m.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

rho_c = rho_micro * 1e-6
lambda_eq = lambda_equiv(rho_c, a, b)
qg_coeff = q_gauss_coeff_per_L(r, a, b, rho_c, is_conductor)  # qgauss/L
E_r = electric_field(r, a, b, rho_c, is_conductor)

if is_conductor:
    lambda_int = 0.0
    lambda_ext = lambda_eq
else:
    lambda_int = None
    lambda_ext = None

# =========================================================
# IMAGEM
# =========================================================
st.markdown('<div class="section-title">Imagem</div>', unsafe_allow_html=True)


def build_svg(a, b, r, rho_c, lambda_eq, E_r, is_conductor, lambda_int, lambda_ext):
    px_per_m = 120.0  # escala fixa

    outer_r_px = max(10.0, b * px_per_m)
    inner_r_px = a * px_per_m
    gauss_r_px = r * px_per_m

    x_front = 270
    x_back = 860
    y_c = 280

    svg_w = int(max(1550, x_back + 330))
    svg_h = int(max(660, y_c + max(outer_r_px, gauss_r_px) + 190))

    sign_color = charge_color(rho_c)
    ext_color = sign_color
    int_color = charge_color(0.0) if is_conductor else sign_color

    ext_fill = soft_fill(ext_color, 0.18)
    gauss_color = "#16a34a"
    field_color = "#111827"
    gray = "#111827"

    # vetor no meio da superfície gaussiana
    # escolhemos o ponto médio ao longo do comprimento, na parte superior da superfície cilíndrica.
    x_vec = (x_front + x_back) / 2
    y_vec = y_c - gauss_r_px

    arrow_len = 85
    if abs(E_r) < 1e-18:
        arrow_svg = ""
        sentido = "nulo"
        field_text = f"E = 0 N/C"
    else:
        outward = E_r > 0
        # na parte superior, sentido "para fora" = para cima
        if outward:
            x2 = x_vec
            y2 = y_vec - arrow_len
            head = f"""
            <line x1="{x2}" y1="{y2}" x2="{x2-6}" y2="{y2+12}" stroke="{field_color}" stroke-width="3"/>
            <line x1="{x2}" y1="{y2}" x2="{x2+6}" y2="{y2+12}" stroke="{field_color}" stroke-width="3"/>
            """
            sentido = "para fora"
        else:
            x2 = x_vec
            y2 = y_vec + arrow_len
            head = f"""
            <line x1="{x2}" y1="{y2}" x2="{x2-6}" y2="{y2-12}" stroke="{field_color}" stroke-width="3"/>
            <line x1="{x2}" y1="{y2}" x2="{x2+6}" y2="{y2-12}" stroke="{field_color}" stroke-width="3"/>
            """
            sentido = "para dentro"

        arrow_svg = f"""
        <line x1="{x_vec}" y1="{y_vec}" x2="{x2}" y2="{y2}" stroke="{field_color}" stroke-width="3"/>
        {head}
        """
        field_text = f"E = {fmt_num(E_r)} N/C"

    # caixas
    info_box_x = 18
    info_box_y = 20
    info_box_w = 370
    info_box_h = 150 if not is_conductor else 190

    field_box_x = 930
    field_box_y = 165
    field_box_w = 360
    field_box_h = 120

    def radius_dimension(x, radius, label):
        """
        Cota de RAIO, não diâmetro:
        vai do centro até a borda superior.
        """
        y1 = y_c
        y2 = y_c - radius
        return f"""
        <line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{gray}" stroke-width="2"/>
        <line x1="{x-8}" y1="{y1}" x2="{x+8}" y2="{y1}" stroke="{gray}" stroke-width="2"/>
        <line x1="{x-8}" y1="{y2}" x2="{x+8}" y2="{y2}" stroke="{gray}" stroke-width="2"/>
        <line x1="{x}" y1="{y2}" x2="{x+7}" y2="{y2+10}" stroke="{gray}" stroke-width="2"/>
        <line x1="{x}" y1="{y2}" x2="{x-7}" y2="{y2+10}" stroke="{gray}" stroke-width="2"/>
        <line x1="{x}" y1="{y1}" x2="{x+7}" y2="{y1-10}" stroke="{gray}" stroke-width="2"/>
        <line x1="{x}" y1="{y1}" x2="{x-7}" y2="{y1-10}" stroke="{gray}" stroke-width="2"/>
        <text x="{x-18}" y="{(y1+y2)/2}" text-anchor="end" dominant-baseline="middle"
              font-size="22" font-weight="700" fill="{gray}">{label}</text>
        """

    dim_x_b = 145
    dim_x_a = 195

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

    # box esquerdo: rho e lambda
    lines = [
        ("ρ =", fmt_num_unit(rho_c, "C/m³"), charge_color(rho_c)),
        ("λ =", fmt_num_unit(lambda_eq, "C/m"), charge_color(lambda_eq)),
    ]
    if is_conductor:
        lines.append(("λint =", fmt_num_unit(lambda_int, "C/m"), charge_color(lambda_int)))
        lines.append(("λext =", fmt_num_unit(lambda_ext, "C/m"), charge_color(lambda_ext)))

    info_text = f'<text x="{info_box_x+14}" y="{info_box_y+30}" font-size="18" font-weight="800" fill="#111827">Densidade e carga por unidade de comprimento</text>'
    for i, (lab, val, col) in enumerate(lines):
        yy = info_box_y + 58 + i * 25
        info_text += f'<text x="{info_box_x+14}" y="{yy}" font-size="18" font-weight="700" fill="{col}">{lab} {val}</text>'

    info_box_svg = f"""
    <rect x="{info_box_x}" y="{info_box_y}" width="{info_box_w}" height="{info_box_h}"
          rx="14" ry="14" fill="#ffffff" stroke="#d1d5db" stroke-width="2"/>
    {info_text}
    """

    field_box_svg = f"""
    <rect x="{field_box_x}" y="{field_box_y}" width="{field_box_w}" height="{field_box_h}"
          rx="14" ry="14" fill="#ffffff" stroke="#d1d5db" stroke-width="2"/>
    <text x="{field_box_x+14}" y="{field_box_y+28}" font-size="18" font-weight="800" fill="#111827">
        Campo elétrico na superfície gaussiana
    </text>
    <text x="{field_box_x+14}" y="{field_box_y+58}" font-size="18" fill="#111827">{field_text}</text>
    <text x="{field_box_x+14}" y="{field_box_y+88}" font-size="18" fill="#111827">Sentido: {sentido}</text>
    """

    params_text = f"""
    <text x="20" y="{svg_h-22}" font-size="18" fill="#111827" font-weight="600">
        a = {fmt_num(a)} m   |   b = {fmt_num(b)} m   |   r = {fmt_num(r)} m
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

            {radius_dimension(dim_x_b, outer_r_px, "b")}
            {radius_dimension(dim_x_a, inner_r_px, "a") if a > 0 else ""}

            {outer_back}
            {outer_rect}
            {inner_svg}
            {outer_front}

            {gauss_svg}
            {arrow_svg}

            {info_box_svg}
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
    rho_c=rho_c,
    lambda_eq=lambda_eq,
    E_r=E_r,
    is_conductor=is_conductor,
    lambda_int=lambda_int,
    lambda_ext=lambda_ext,
)

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

                q<sub>gauss</sub> = ρ · π (b² − a²) L<br><br>

                q<sub>gauss</sub> = ({fmt_html(rho_c, "C/m³")}) · π · ({fmt_html(b**2, "m²")} − {fmt_html(a**2, "m²")}) · L<br><br>

                <strong>q<sub>gauss</sub> = {fmt_html(qg_coeff, "C/m")} · L</strong>
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

                    ρ = q<sub>gauss</sub> / V<sub>r</sub>, sendo V o volume.<br><br>

                    ρ = q<sub>gauss</sub> / [π (r² − a²) L]<br><br>

                    q<sub>gauss</sub> = ρ π (r² − a²) L<br><br>

                    q<sub>gauss</sub> = ({fmt_html(rho_c, "C/m³")}) · π · ({fmt_html(r**2, "m²")} − {fmt_html(a**2, "m²")}) · L<br><br>

                    <strong>q<sub>gauss</sub> = {fmt_html(qg_coeff, "C/m")} · L</strong>
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
    coeff_area = 2 * math.pi * r
    st.markdown(
        f"""
        <div class="formula-card black-text">
            <div class="equation">
                A = 2πrL<br><br>
                A = 2π · {fmt_html(r, "m")} · L<br><br>
                <strong>A = {fmt_html(coeff_area, "m")} · L</strong>
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
        field_html = f"""
        <div class="equation">
            Lei de Gauss no caso de simetria: campo constante E em toda superfície gaussiana
            e sempre paralelo ao vetor área.<br><br>

            E A = q<sub>gauss</sub> / ε<sub>0</sub><br><br>

            E · (2πrL) = [ρπ(b² − a²)L] / ε<sub>0</sub><br><br>

            E = [ρπ(b² − a²)L] / [2πrLε<sub>0</sub>]<br><br>

            E = ρ (b² − a²) / (2rε<sub>0</sub>)<br><br>

            E = ({fmt_html(rho_c, "C/m³")}) · ({fmt_html(b**2, "m²")} − {fmt_html(a**2, "m²")})
            / [2 · {fmt_html(r, "m")} · 8,8 × 10<sup>-12</sup> C²/N·m²]<br><br>

            <strong>E = {fmt_html(E_r, "N/C")}</strong>
        </div>
        """
    elif r < a:
        field_html = """
        <div class="equation">
            E A = q<sub>gauss</sub> / ε<sub>0</sub><br><br>
            Como q<sub>gauss</sub> = 0, então:<br><br>
            <strong>E = 0 N/C</strong>
        </div>
        """
    else:
        if is_conductor:
            field_html = """
            <div class="equation">
                E A = q<sub>gauss</sub> / ε<sub>0</sub><br><br>
                Como q<sub>gauss</sub> = 0, então:<br><br>
                <strong>E = 0 N/C</strong>
            </div>
            """
        else:
            field_html = f"""
            <div class="equation">
                Lei de Gauss no caso de simetria: campo constante E em toda superfície gaussiana
                e sempre paralelo ao vetor área.<br><br>

                E A = q<sub>gauss</sub> / ε<sub>0</sub><br><br>

                E · (2πrL) = [ρπ(r² − a²)L] / ε<sub>0</sub><br><br>

                E = [ρπ(r² − a²)L] / [2πrLε<sub>0</sub>]<br><br>

                E = ρ (r² − a²) / (2rε<sub>0</sub>)<br><br>

                E = ({fmt_html(rho_c, "C/m³")}) · ({fmt_html(r**2, "m²")} − {fmt_html(a**2, "m²")})
                / [2 · {fmt_html(r, "m")} · 8,8 × 10<sup>-12</sup> C²/N·m²]<br><br>

                <strong>E = {fmt_html(E_r, "N/C")}</strong>
            </div>
            """

    st.markdown(
        f"""
        <div class="formula-card black-text">
            {field_html}
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

r_max_graph = 2.5
rr = np.linspace(0.001, r_max_graph, 900)
EE = np.array([electric_field(float(rv), a, b, rho_c, is_conductor) for rv in rr])

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
# RODAPÉ
# =========================================================
st.markdown(
    """
    <div class="white-card black-text">
        <div class="small-note">
            <strong>Observações desta versão:</strong>
            <ul>
                <li>O slider de <strong>L</strong> foi removido.</li>
                <li>O campo elétrico agora é calculado a partir de <strong>ρ</strong>.</li>
                <li>Para <strong>condutor</strong>, a carga fica toda na superfície externa.</li>
                <li>As cotas <strong>a</strong> e <strong>b</strong> na imagem representam <strong>raios</strong>.</li>
                <li>O vetor do campo foi movido para o meio da superfície gaussiana.</li>
                <li>O gráfico continua com ajuste automático de eixos, mas sem permitir zoom manual.</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
