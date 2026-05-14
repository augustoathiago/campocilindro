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
    """Formata número sem usar e/E."""
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


def to_latex_num(x: float, digits: int = 4) -> str:
    """Número formatado para LaTeX sem e/E."""
    if abs(x) < 1e-15:
        return "0"

    ax = abs(x)
    if 1e-3 <= ax < 1e4:
        return f"{x:.{digits}f}".rstrip("0").rstrip(".")

    exp = int(math.floor(math.log10(ax)))
    mant = x / (10 ** exp)
    mant_str = f"{mant:.{digits}f}".rstrip("0").rstrip(".")
    return rf"{mant_str}\times 10^{{{exp}}}"


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
# FÍSICA - CILINDRO ISOLANTE
# =========================================================
def q_gauss_coeff_per_L(r: float, a: float, b: float, rho_c_m3: float) -> float:
    """
    Retorna q_gauss/L em C/m.

    Casos:
      - r < a        -> 0
      - a <= r < b   -> ρ π (r² - a²)
      - r >= b       -> ρ π (b² - a²)
    """
    if r < 0:
        return 0.0

    if r < a:
        return 0.0

    if a <= r < b:
        return rho_c_m3 * math.pi * (r**2 - a**2)

    return rho_c_m3 * math.pi * (b**2 - a**2)


def electric_field(r: float, a: float, b: float, rho_c_m3: float) -> float:
    """
    Campo elétrico assinado:
      positivo -> para fora
      negativo -> para dentro

    q_gauss = [coef] * L
    A = 2πrL
    => L cancela
    """
    if r <= 0:
        return 0.0

    q_coeff = q_gauss_coeff_per_L(r, a, b, rho_c_m3)  # C/m
    return q_coeff / (2 * math.pi * r * EPSILON_0)


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

        .slider-card {
            background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%);
            border: 2px solid #60a5fa;
            border-radius: 18px;
            padding: 1rem 1rem 0.5rem 1rem;
            margin-top: 0.5rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.08);
        }

        .formula-shell {
            background: #ffffff;
            border: 1px solid #d1d5db;
            border-left: 5px solid #2563eb;
            border-radius: 14px;
            padding: 1rem 1rem;
            color: #111827 !important;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        }

        .small-note {
            color: #4b5563;
            font-size: 0.95rem;
        }

        .black-text * {
            color: #111827 !important;
        }

        [data-testid="stWidgetLabel"] p,
        [data-testid="stMarkdownContainer"] p,
        .stSlider label {
            color: #111827 !important;
        }

        .stSlider > div > div {
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
    st.markdown('<div class="main-subtitle">Estude o campo elétrico de um cilindro longo isolante.</div>', unsafe_allow_html=True)

# =========================================================
# ESTADO
# =========================================================
if "a" not in st.session_state:
    st.session_state.a = 0.45

if "b" not in st.session_state:
    st.session_state.b = 1.0

if "rho_micro" not in st.session_state:
    st.session_state.rho_micro = 6.0

if "r" not in st.session_state:
    st.session_state.r = 0.8

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

    b_max = 2.0
    b_min = round(min(a + 0.5, b_max), 2)

    if abs(b_min - b_max) < 1e-9:
        b = 2.0
        st.session_state.b = 2.0

        st.slider(
            "Raio externo b do cilindro (m)",
            min_value=0.0,
            max_value=2.0,
            value=2.0,
            step=0.05,
            disabled=True,
            key="b_locked_visible",
            help="b ficou travado em 2,0 m porque deve satisfazer b ≥ a + 0,5 m."
        )
    else:
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

    st.markdown('</div>', unsafe_allow_html=True)

rho_c = rho_micro * 1e-6
qg_coeff = q_gauss_coeff_per_L(r, a, b, rho_c)
E_r = electric_field(r, a, b, rho_c)

# =========================================================
# IMAGEM
# =========================================================
st.markdown('<div class="section-title">Imagem</div>', unsafe_allow_html=True)


def build_svg(a, b, r, rho_c, E_r):
    px_per_m = 120.0

    outer_r_px = max(10.0, b * px_per_m)
    inner_r_px = a * px_per_m
    gauss_r_px = r * px_per_m

    x_front = 270
    x_back = 860
    y_c = 280

    svg_w = int(max(1550, x_back + 380))
    svg_h = int(max(680, y_c + max(outer_r_px, gauss_r_px) + 200))

    sign_color = charge_color(rho_c)
    ext_color = sign_color
    int_color = sign_color
    ext_fill = soft_fill(ext_color, 0.18)

    gauss_color = "#16a34a"
    gray = "#111827"

    def radius_dimension(x, radius, label):
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

    sentido = "para fora" if E_r > 0 else "para dentro" if E_r < 0 else "nulo"
    field_text = f"E = {fmt_num(E_r)} N/C"

    field_box_x = 930
    field_box_y = 145
    field_box_w = 300
    field_box_h = 125

    field_box_svg = f"""
    <rect x="{field_box_x}" y="{field_box_y}" width="{field_box_w}" height="{field_box_h}"
          rx="14" ry="14" fill="#ffffff" stroke="#d1d5db" stroke-width="2"/>
    <text x="{field_box_x+14}" y="{field_box_y+28}" font-size="18" font-weight="800" fill="#111827">
        Campo elétrico
    </text>
    <text x="{field_box_x+14}" y="{field_box_y+52}" font-size="18" font-weight="800" fill="#111827">
        na superfície gaussiana
    </text>
    <text x="{field_box_x+14}" y="{field_box_y+82}" font-size="18" fill="#111827">{field_text}</text>
    <text x="{field_box_x+14}" y="{field_box_y+108}" font-size="18" fill="#111827">Sentido: {sentido}</text>
    """

    charge_box_x = field_box_x
    charge_box_y = field_box_y + field_box_h + 18
    charge_box_w = 300
    charge_box_h = 85

    charge_box_svg = f"""
    <rect x="{charge_box_x}" y="{charge_box_y}" width="{charge_box_w}" height="{charge_box_h}"
          rx="14" ry="14" fill="#ffffff" stroke="#d1d5db" stroke-width="2"/>
    <text x="{charge_box_x+14}" y="{charge_box_y+28}" font-size="18" font-weight="800" fill="#111827">
        Densidade de carga
    </text>
    <text x="{charge_box_x+14}" y="{charge_box_y+60}" font-size="18" font-weight="700" fill="{charge_color(rho_c)}">
        ρ = {fmt_num_unit(rho_c, "C/m³")}
    </text>
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

            {field_box_svg}
            {charge_box_svg}

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
    E_r=E_r,
)

components.html(svg_html, height=svg_height, scrolling=True)

# =========================================================
# LEI DE GAUSS
# =========================================================
st.markdown('<div class="section-title">Lei de Gauss</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="formula-shell">
        <p><strong>Lei de Gauss:</strong></p>
        <p style="font-size:1.08rem; margin-bottom:0.7rem;">
            <strong>Φ = ∮ E · dA = q<sub>gauss</sub> / ε<sub>0</sub></strong>
        </p>
        <ul style="line-height:1.8; margin-top:0.3rem;">
            <li><strong>Φ</strong>: fluxo elétrico</li>
            <li><strong>E</strong>: campo elétrico</li>
            <li><strong>A</strong>: área da superfície gaussiana</li>
            <li><strong>q<sub>gauss</sub></strong>: carga dentro da superfície gaussiana</li>
            <li><strong>ε<sub>0</sub></strong> = 8,8 × 10<sup>-12</sup> C²/(N·m²)</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CARGA q_gauss
# =========================================================
st.markdown('<div class="section-title">Carga q<sub>gauss</sub> contida na superfície gaussiana</div>', unsafe_allow_html=True)
st.markdown('<div class="formula-shell">', unsafe_allow_html=True)

if r >= b:
    st.markdown("**Caso de superfície gaussiana fora do cilindro, englobando toda carga:**")

    st.latex(r"\rho=\frac{q_{gauss}}{V_r}")

    if abs(a) < 1e-12:
        st.latex(r"V_r=\pi\,b^2\,L")
        st.write("")
        st.latex(r"\rho=\frac{q_{gauss}}{\pi\,b^2\,L}")
        st.write("")
        st.latex(r"q_{gauss}=\rho\,\pi\,b^2\,L")
        st.latex(
            rf"q_{{gauss}}=({to_latex_num(rho_c)})\,\pi\,({to_latex_num(b**2)})\,L"
        )
    else:
        st.latex(r"V_r=\pi\,(b^2-a^2)\,L")
        st.latex(r"\rho=\frac{q_{gauss}}{\pi\,(b^2-a^2)\,L}")
        st.latex(r"q_{gauss}=\rho\,\pi\,(b^2-a^2)\,L")
        st.latex(
            rf"q_{{gauss}}=({to_latex_num(rho_c)})\,\pi\,({to_latex_num(b**2)}-{to_latex_num(a**2)})\,L"
        )

    st.latex(
        rf"q_{{gauss}}=({to_latex_num(qg_coeff)}\ \mathrm{{C/m}})\,L"
    )

elif r < a:
    st.markdown("**Caso da superfície gaussiana dentro da cavidade do cilindro, sem englobar carga alguma:**")
    st.latex(r"q_{gauss}=0")

else:
    st.markdown("**Caso da superfície gaussiana entre as superfícies do cilindro, englobando apenas uma parcela da carga:**")

    if abs(a) < 1e-12:
        st.latex(r"\rho=\frac{q_{gauss}}{V_r}")
        st.latex(r"V_r=\pi\,r^2\,L")
        st.latex(r"\rho=\frac{q_{gauss}}{\pi r^2L}")
        st.latex(r"q_{gauss}=\rho\,\pi\,r^2\,L")
        st.latex(
            rf"q_{{gauss}}=({to_latex_num(rho_c)})\,\pi\,({to_latex_num(r**2)})\,L"
        )
    else:
        st.latex(r"\rho=\frac{q_{gauss}}{V_r}")
        st.latex(r"V_r=\pi\,(r^2-a^2)\,L")
        st.latex(r"\rho=\frac{q_{gauss}}{\pi\,(r^2-a^2)\,L}")
        st.latex(r"q_{gauss}=\rho\,\pi\,(r^2-a^2)\,L")
        st.latex(
            rf"q_{{gauss}}=({to_latex_num(rho_c)})\,\pi\,({to_latex_num(r**2)}-{to_latex_num(a**2)})\,L"
        )

    st.latex(
        rf"q_{{gauss}}=({to_latex_num(qg_coeff)}\ \mathrm{{C/m}})\,L"
    )

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# ÁREA
# =========================================================
st.markdown('<div class="section-title">Área da superfície gaussiana</div>', unsafe_allow_html=True)
st.markdown('<div class="formula-shell">', unsafe_allow_html=True)

if r > 0:
    area_coeff = 2 * math.pi * r
    st.latex(r"A=2\pi rL")
    st.latex(rf"A=2\pi\,({to_latex_num(r)})\,L")
    st.latex(rf"A=({to_latex_num(area_coeff)}\ \mathrm{{m}})\,L")
else:
    st.markdown("Para \(r=0\), a área da superfície gaussiana cilíndrica não é definida.")

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# CAMPO ELÉTRICO
# =========================================================
st.markdown('<div class="section-title">Campo elétrico</div>', unsafe_allow_html=True)
st.markdown('<div class="formula-shell">', unsafe_allow_html=True)

if r > 0:
    if r >= b:
        st.latex(r"E\,A=\frac{q_{gauss}}{\varepsilon_0}")

        if abs(a) < 1e-12:
            st.latex(r"E\,(2\pi rL)=\frac{\rho\pi b^2L}{\varepsilon_0}")
            st.latex(r"E=\frac{\rho\pi b^2L}{2\pi rL\,\varepsilon_0}")
            st.latex(r"E=\frac{\rho\,b^2}{2r\,\varepsilon_0}")
            st.latex(
                rf"E=\frac{{({to_latex_num(rho_c)})\,({to_latex_num(b**2)})}}{{2\,({to_latex_num(r)})\,(8.8\times10^{{-12}})}}"
            )
        else:
            st.latex(r"E\,(2\pi rL)=\frac{\rho\pi(b^2-a^2)L}{\varepsilon_0}")
            st.latex(r"E=\frac{\rho\pi(b^2-a^2)L}{2\pi rL\,\varepsilon_0}")
            st.latex(r"E=\frac{\rho\,(b^2-a^2)}{2r\,\varepsilon_0}")
            st.latex(
                rf"E=\frac{{({to_latex_num(rho_c)})\,({to_latex_num(b**2)}-{to_latex_num(a**2)})}}{{2\,({to_latex_num(r)})\,(8.8\times10^{{-12}})}}"
            )

        st.latex(rf"E={to_latex_num(E_r)}\ \mathrm{{N/C}}")

    elif r < a:
        st.latex(r"q_{gauss}=0")
        st.latex(r"E=0")

    else:
        st.latex(r"E\,A=\frac{q_{gauss}}{\varepsilon_0}")

        if abs(a) < 1e-12:
            st.latex(r"E\,(2\pi rL)=\frac{\rho\pi r^2L}{\varepsilon_0}")
            st.latex(r"E=\frac{\rho\pi r^2L}{2\pi rL\,\varepsilon_0}")
            st.latex(r"E=\frac{\rho\,r}{2\,\varepsilon_0}")
            st.latex(
                rf"E=\frac{{({to_latex_num(rho_c)})\,({to_latex_num(r)})}}{{2\,(8.8\times10^{{-12}})}}"
            )
        else:
            st.latex(r"E\,(2\pi rL)=\frac{\rho\pi(r^2-a^2)L}{\varepsilon_0}")
            st.latex(r"E=\frac{\rho\pi(r^2-a^2)L}{2\pi rL\,\varepsilon_0}")
            st.latex(r"E=\frac{\rho\,(r^2-a^2)}{2r\,\varepsilon_0}")
            st.latex(
                rf"E=\frac{{({to_latex_num(rho_c)})\,({to_latex_num(r**2)}-{to_latex_num(a**2)})}}{{2\,({to_latex_num(r)})\,(8.8\times10^{{-12}})}}"
            )

        st.latex(rf"E={to_latex_num(E_r)}\ \mathrm{{N/C}}")

    st.markdown(
        f"""
        <div class="small-note" style="margin-top:0.4rem;">
            Sentido do campo neste ponto:
            <strong>{'para fora' if E_r > 0 else 'para dentro' if E_r < 0 else 'nulo'}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("Para \(r=0\), o app exibe o campo como 0 por convenção visual no eixo.")

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# GRÁFICO
# =========================================================
st.markdown('<div class="section-title">Gráfico</div>', unsafe_allow_html=True)

# IMPORTANTE:
# O erro do print estava em fig = go.Figure()
# Nesta versão, go está corretamente importado no topo:
# import plotly.graph_objects as go

r_max_graph = 2.5
rr = np.linspace(0.001, r_max_graph, 900)
EE = np.array([electric_field(float(rv), a, b, rho_c) for rv in rr], dtype=float)

# Evita problemas quando todos os valores forem zero
if EE.size == 0:
    EE = np.array([0.0])

y_abs = float(np.nanmax(np.abs(EE)))
if not np.isfinite(y_abs) or y_abs < 1e-9:
    y_abs = 1.0

y_margin = 0.12 * y_abs

# Criação correta da figura Plotly
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
