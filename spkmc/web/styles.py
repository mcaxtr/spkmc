"""
Design system for SPKMC web interface.

Clean, professional aesthetic inspired by modern SaaS dashboards.
Soft teal accents, dark sidebar, generous whitespace, refined typography.
"""

import base64
import textwrap

# SVG Icons - Simple, professional stroke icons (Feather/Lucide style)
ICONS = {
    "flask": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round"><path d="M9 3h6M9 3v9l-5 9h16l-5-9V3"/></svg>'
    ),
    "file": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 '
        '2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>'
    ),
    "check": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
    ),
    "clock": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>'
        '<polyline points="12 6 12 12 16 14"/></svg>'
    ),
    "settings": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round"><circle cx="12" cy="12" r="3"/>'
        '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 '
        "2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 "
        "1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 "
        "1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 "
        "0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 "
        "0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 "
        "4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 "
        "0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 "
        "1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 "
        "1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 "
        "0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 "
        "1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 "
        '1z"/></svg>'
    ),
}

# Refined color palette - soft teals and clean neutrals
COLORS = {
    # Primary palette - soft teal/sage
    "teal_700": "#1E5F55",
    "teal_600": "#2D7A6E",
    "teal_500": "#4A9E8E",
    "teal_400": "#5FB5A6",
    "teal_300": "#8ECFC3",
    "teal_100": "#E8F5F3",
    "teal_50": "#F0F9F7",
    # Neutrals - clean grays
    "gray_950": "#0B0F19",
    "gray_900": "#111827",
    "gray_800": "#1F2937",
    "gray_700": "#374151",
    "gray_600": "#4B5563",
    "gray_500": "#6B7280",
    "gray_400": "#9CA3AF",
    "gray_300": "#D1D5DB",
    "gray_200": "#E5E7EB",
    "gray_100": "#F3F4F6",
    "gray_50": "#F9FAFB",
    # Background
    "bg_primary": "#F7F8FA",
    "bg_secondary": "#FAFBFC",
    # White
    "white": "#FFFFFF",
    # Status colors - muted and professional
    "success": "#10B981",
    "success_bg": "#D1FAE5",
    "warning": "#F59E0B",
    "warning_bg": "#FEF3C7",
    "error": "#EF4444",
    "error_bg": "#FEE2E2",
    "info": "#3B82F6",
    "info_bg": "#DBEAFE",
}

FONTS = {
    "body": "'Plus Jakarta Sans', 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif",
    "mono": "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
}


def _dedent(html: str) -> str:
    """Strip leading whitespace from HTML to prevent Markdown code-block rendering."""
    return textwrap.dedent(html).strip()


def _svg_data_uri(svg: str) -> str:
    """Convert raw SVG string to a CSS-safe base64 data URI."""
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'url("data:image/svg+xml;base64,{encoded}")'


def get_global_styles() -> str:
    """
    Returns comprehensive CSS for the entire application.
    Clean, professional aesthetic with soft teal accents and dark sidebar.
    """
    # SVG data URIs for sidebar nav icons
    experiments_icon_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" '
        'viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.5)" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M9 3h6M9 3v9l-5 9h16l-5-9V3"/></svg>'
    )
    experiments_icon_active_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" '
        'viewBox="0 0 24 24" fill="none" stroke="#5FB5A6" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M9 3h6M9 3v9l-5 9h16l-5-9V3"/></svg>'
    )
    settings_icon_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" '
        'viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.5)" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 '
        "2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 "
        "1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 "
        "19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 "
        "1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3"
        "a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82"
        "l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 "
        "0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 "
        "0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 "
        "1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 "
        '1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>'
    )
    settings_icon_active_svg = settings_icon_svg.replace(
        'stroke="rgba(255,255,255,0.5)"', 'stroke="#5FB5A6"'
    )

    exp_icon = _svg_data_uri(experiments_icon_svg)
    exp_icon_active = _svg_data_uri(experiments_icon_active_svg)
    set_icon = _svg_data_uri(settings_icon_svg)
    set_icon_active = _svg_data_uri(settings_icon_active_svg)

    return _dedent(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
    --teal-700: {COLORS['teal_700']};
    --teal-600: {COLORS['teal_600']};
    --teal-500: {COLORS['teal_500']};
    --teal-400: {COLORS['teal_400']};
    --teal-300: {COLORS['teal_300']};
    --teal-100: {COLORS['teal_100']};
    --teal-50: {COLORS['teal_50']};
    --gray-950: {COLORS['gray_950']};
    --gray-900: {COLORS['gray_900']};
    --gray-800: {COLORS['gray_800']};
    --gray-700: {COLORS['gray_700']};
    --gray-600: {COLORS['gray_600']};
    --gray-500: {COLORS['gray_500']};
    --gray-400: {COLORS['gray_400']};
    --gray-300: {COLORS['gray_300']};
    --gray-200: {COLORS['gray_200']};
    --gray-100: {COLORS['gray_100']};
    --gray-50: {COLORS['gray_50']};
    --bg-primary: {COLORS['bg_primary']};
    --white: {COLORS['white']};
    --success: {COLORS['success']};
    --warning: {COLORS['warning']};
    --error: {COLORS['error']};
    --info: {COLORS['info']};
    --font-body: {FONTS['body']};
    --font-mono: {FONTS['mono']};
    --shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.06), 0 1px 2px -1px rgba(0, 0, 0, 0.06);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
    --radius-xl: 16px;
    --transition-fast: 120ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-base: 200ms cubic-bezier(0.4, 0, 0.2, 1);
}}

/* ── Base ────────────────────────────────────── */
.stApp {{
    background: var(--bg-primary);
    font-family: var(--font-body);
}}

/* Hide Streamlit chrome (keep sidebar collapse button visible) */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none; }}
[data-testid="stSidebarNav"] {{ display: none; }}

/* ── Sidebar ─────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {COLORS['gray_950']} 0%, {COLORS['gray_900']} 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
    width: 260px !important;
}}
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div {{
    padding-top: 0 !important;
}}
section[data-testid="stSidebar"] > div {{
    padding: 1rem 1.25rem 1.5rem 1.25rem;
}}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: rgba(255, 255, 255, 0.9) !important;
}}

/* Sidebar buttons – default state */
section[data-testid="stSidebar"] .stButton > button {{
    background: transparent;
    border: none;
    color: rgba(255, 255, 255, 0.55);
    border-radius: var(--radius-md);
    padding: 0.625rem 0.75rem 0.625rem 2.5rem;
    font-family: var(--font-body);
    font-weight: 500;
    font-size: 0.875rem;
    text-align: left;
    transition: all var(--transition-base);
    margin-bottom: 2px;
    position: relative;
    box-shadow: none;
}}
section[data-testid="stSidebar"] .stButton > button::before {{
    content: "";
    position: absolute;
    left: 0.75rem;
    top: 50%;
    transform: translateY(-50%);
    width: 18px;
    height: 18px;
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    opacity: 0.7;
    transition: opacity var(--transition-base);
}}

/* Sidebar button hover */
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255, 255, 255, 0.06);
    color: rgba(255, 255, 255, 0.85);
    transform: none;
    box-shadow: none;
}}
section[data-testid="stSidebar"] .stButton > button:hover::before {{
    opacity: 1;
}}

/* Sidebar button active/primary */
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background: rgba(95, 181, 166, 0.1);
    color: var(--teal-400);
    border-left: 2px solid var(--teal-400);
    padding-left: calc(2.5rem - 2px);
    box-shadow: none;
}}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
    background: rgba(95, 181, 166, 0.15);
    color: var(--teal-300);
    box-shadow: none;
}}

/* Icon injection: Experiments button */
section[data-testid="stSidebar"] .st-key-nav_experiments button::before {{
    background-image: {exp_icon};
}}
section[data-testid="stSidebar"] .st-key-nav_experiments button[kind="primary"]::before {{
    background-image: {exp_icon_active};
    opacity: 1;
}}

/* Icon injection: Preferences button */
section[data-testid="stSidebar"] .st-key-nav_settings button::before {{
    background-image: {set_icon};
}}
section[data-testid="stSidebar"] .st-key-nav_settings button[kind="primary"]::before {{
    background-image: {set_icon_active};
    opacity: 1;
}}

/* Sidebar collapse button */
[data-testid="stSidebarCollapsedControl"] {{
    color: var(--gray-400);
}}

/* Sidebar version footer pinned to bottom */
.sidebar-version-footer {{
    position: fixed;
    bottom: 1.5rem;
    left: 0;
    width: 260px;
    padding: 0 1.25rem;
    box-sizing: border-box;
    z-index: 100;
}}

/* ── Typography ──────────────────────────────── */
h1, h2, h3, h4, h5, h6 {{
    font-family: var(--font-body) !important;
    letter-spacing: -0.02em;
}}
h1 {{
    font-size: 1.875rem !important;
    font-weight: 800 !important;
    color: var(--gray-900) !important;
    line-height: 1.2 !important;
    margin-bottom: 0.25rem !important;
}}
h2 {{
    font-size: 1.375rem !important;
    font-weight: 700 !important;
    color: var(--gray-800) !important;
    line-height: 1.3 !important;
    margin-bottom: 0.75rem !important;
}}
h3 {{
    font-size: 1.125rem !important;
    font-weight: 600 !important;
    color: var(--gray-800) !important;
    margin-bottom: 0.5rem !important;
}}
p {{
    font-family: var(--font-body);
    font-size: 0.938rem;
    line-height: 1.65;
    color: var(--gray-600);
}}
.stCaption, [data-testid="stCaptionContainer"] {{
    font-family: var(--font-body) !important;
    font-size: 0.813rem !important;
    color: var(--gray-500) !important;
}}

/* ── Buttons (main content) ──────────────────── */
/* Force inner elements (Streamlit wraps text in <p>) to inherit button color */
.stButton > button p,
.stButton > button span,
.stButton > button div,
[data-testid="stDialog"] button p,
[data-testid="stDialog"] button span,
[data-testid="stDialog"] button div {{
    color: inherit !important;
}}
.stMainBlockContainer .stButton > button,
.stMainBlockContainer button[data-testid="stBaseButton-primary"] {{
    background: var(--teal-600) !important;
    color: var(--white) !important;
    border: none !important;
    border-radius: var(--radius-md);
    padding: 0.5rem 1.125rem;
    font-family: var(--font-body);
    font-weight: 600;
    font-size: 0.813rem;
    box-shadow: var(--shadow-xs);
    transition: all var(--transition-base);
    letter-spacing: 0.01em;
}}
.stMainBlockContainer .stButton > button:hover,
.stMainBlockContainer button[data-testid="stBaseButton-primary"]:hover {{
    background: var(--teal-700) !important;
    color: var(--white) !important;
    box-shadow: var(--shadow-sm);
    transform: translateY(-1px);
}}
.stMainBlockContainer .stButton > button:active {{
    transform: translateY(0);
    box-shadow: var(--shadow-xs);
}}
.stMainBlockContainer .stButton > button[kind="secondary"],
.stMainBlockContainer button[data-testid="stBaseButton-secondary"] {{
    background: var(--white) !important;
    color: var(--gray-700) !important;
    border: 1px solid var(--gray-300) !important;
    box-shadow: var(--shadow-xs);
}}
.stMainBlockContainer .stButton > button[kind="secondary"]:hover,
.stMainBlockContainer button[data-testid="stBaseButton-secondary"]:hover {{
    background: var(--gray-50) !important;
    border-color: var(--gray-400) !important;
    color: var(--gray-900) !important;
}}
/* Dialog/modal buttons -- green for constructive actions */
[data-testid="stDialog"] button[data-testid="stBaseButton-primary"],
[data-testid="stDialog"] .stButton button[kind="primary"],
[data-testid="stDialog"] [data-testid="stBaseButton-primary"] {{
    background: {COLORS['success']} !important;
    color: var(--white) !important;
    border: none !important;
    box-shadow: var(--shadow-xs) !important;
}}
[data-testid="stDialog"] button[data-testid="stBaseButton-primary"] p,
[data-testid="stDialog"] button[data-testid="stBaseButton-primary"] span,
[data-testid="stDialog"] .stButton button[kind="primary"] p,
[data-testid="stDialog"] .stButton button[kind="primary"] span,
[data-testid="stDialog"] [data-testid="stBaseButton-primary"] p,
[data-testid="stDialog"] [data-testid="stBaseButton-primary"] span {{
    color: var(--white) !important;
}}
[data-testid="stDialog"] button[data-testid="stBaseButton-primary"]:hover,
[data-testid="stDialog"] .stButton button[kind="primary"]:hover,
[data-testid="stDialog"] [data-testid="stBaseButton-primary"]:hover {{
    background: #059669 !important;
    color: var(--white) !important;
    box-shadow: var(--shadow-sm) !important;
}}

/* ── Form inputs ─────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {{
    border: 1px solid var(--gray-300);
    border-radius: var(--radius-md);
    padding: 0.563rem 0.875rem;
    font-family: var(--font-body);
    font-size: 0.875rem;
    transition: all var(--transition-fast);
    background: var(--white);
    color: var(--gray-900);
}}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
    border-color: var(--teal-500);
    box-shadow: 0 0 0 3px {COLORS['teal_100']};
    outline: none;
}}
.stSelectbox > div > div {{
    font-family: var(--font-body);
    font-size: 0.875rem;
}}

/* ── Dividers ────────────────────────────────── */
.stDivider {{
    border-color: var(--gray-200);
}}

/* ── Metrics ─────────────────────────────────── */
[data-testid="stMetric"] {{
    background: var(--white);
    border-radius: var(--radius-lg);
    padding: 1rem 1.25rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--gray-200);
}}
[data-testid="stMetricLabel"] {{
    font-family: var(--font-body) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--gray-500) !important;
}}
[data-testid="stMetricValue"] {{
    font-family: var(--font-body) !important;
    font-weight: 700 !important;
    color: var(--gray-900) !important;
}}

/* ── Expanders ───────────────────────────────── */
.streamlit-expanderHeader {{
    font-family: var(--font-body);
    font-weight: 600;
    font-size: 0.938rem;
    color: var(--gray-800);
    background: var(--gray-50);
    border-radius: var(--radius-md);
}}

/* ── Status badges (CSS classes) ─────────────── */
.status-badge {{
    display: inline-block;
    padding: 0.2rem 0.625rem;
    border-radius: 99px;
    font-family: var(--font-body);
    font-size: 0.688rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    line-height: 1.5;
}}
.status-pending {{
    background: {COLORS['gray_100']};
    color: {COLORS['gray_600']};
}}
.status-running {{
    background: {COLORS['info_bg']};
    color: {COLORS['info']};
}}
.status-completed {{
    background: {COLORS['success_bg']};
    color: {COLORS['success']};
}}
.status-failed {{
    background: {COLORS['error_bg']};
    color: {COLORS['error']};
}}
.status-created {{
    background: {COLORS['teal_100']};
    color: {COLORS['teal_500']};
}}
.status-edited {{
    background: {COLORS['teal_100']};
    color: {COLORS['teal_500']};
}}

/* ── Tabs ────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 2px solid var(--gray-200);
}}
.stTabs [data-baseweb="tab"] {{
    font-family: var(--font-body);
    font-weight: 500;
    font-size: 0.875rem;
    color: var(--gray-500);
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    padding: 0.75rem 1.25rem;
}}
.stTabs [aria-selected="true"] {{
    color: var(--teal-600) !important;
    border-bottom-color: var(--teal-500) !important;
    font-weight: 600;
}}

/* ── Dialogs / Modals ────────────────────────── */
[data-testid="stDialog"] [role="dialog"] {{
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-lg);
    border: 1px solid var(--gray-200);
    height: 90vh;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
}}
/* Propagate flex layout through Streamlit wrapper divs to the content block */
[data-testid="stDialog"] [role="dialog"] > [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stDialog"] [role="dialog"] > [data-testid="stVerticalBlockBorderWrapper"] > div,
[data-testid="stDialog"] [role="dialog"] [data-testid="stVerticalBlock"] {{
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
}}
/* Default: children don't shrink so scrolling works normally */
[data-testid="stDialog"] [role="dialog"] [data-testid="stVerticalBlock"] > div {{
    flex-shrink: 0;
}}
/* Action button containers pin to the bottom of the dialog */
[data-testid="stDialog"] [class*="st-key-modal_actions"] {{
    margin-top: auto !important;
}}

/* ── Animations ──────────────────────────────── */
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
.fade-in-up {{ animation: fadeInUp 0.35s ease-out forwards; }}

@keyframes pulse-dot {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
}}

/* ── Scenario progress bar ──────────────────── */
.scenario-progress-bar {{
    width: 100%;
    height: 4px;
    background: {COLORS['teal_100']};
    border-radius: 2px;
    overflow: hidden;
}}
.scenario-progress-fill {{
    height: 100%;
    background: {COLORS['teal_500']};
    border-radius: 2px;
    transition: width 0.5s ease;
}}

/* ── Dataframes ──────────────────────────────── */
.stDataFrame {{
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}}

/* ── Download buttons ────────────────────────── */
.stDownloadButton > button {{
    background: var(--white);
    color: var(--gray-700);
    border: 1px solid var(--gray-300);
    font-size: 0.813rem;
}}
.stDownloadButton > button:hover {{
    background: var(--gray-50);
    border-color: var(--gray-400);
}}

/* ── Clickable experiment cards ─────────────── */
[class*="st-key-exp_card_"] {{
    position: relative;
    cursor: pointer;
}}
[class*="st-key-exp_card_"]:hover .exp-card {{
    box-shadow: var(--shadow-md);
    border-color: var(--teal-300);
}}
/* Stretch the element container (button wrapper) over the card */
[class*="st-key-exp_card_"] [class*="st-key-exp_btn_"] {{
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    z-index: 10;
}}
[class*="st-key-exp_card_"] [class*="st-key-exp_btn_"] .stButton {{
    width: 100%;
    height: 100%;
}}
[class*="st-key-exp_card_"] [class*="st-key-exp_btn_"] .stButton > button {{
    width: 100% !important;
    height: 100% !important;
    opacity: 0 !important;
    cursor: pointer !important;
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    transform: none !important;
}}
[class*="st-key-exp_card_"] [class*="st-key-exp_btn_"] .stButton > button:hover {{
    transform: none !important;
    box-shadow: none !important;
}}

/* ── Clickable scenario cards ────────────── */
/* Stretch columns so action buttons fill the full card height */
[class*="st-key-sc_card_"] [data-testid="stHorizontalBlock"] {{
    align-items: stretch;
}}
[class*="st-key-sc_card_"] [data-testid="stColumn"] {{
    display: flex;
    flex-direction: column;
}}

[class*="st-key-sc_card_"] {{
    position: relative;
    cursor: pointer;
}}
[class*="st-key-sc_card_"]:hover .scenario-card {{
    box-shadow: var(--shadow-md);
    border-color: var(--teal-300);
}}
[class*="st-key-sc_card_"] [class*="st-key-sc_btn_"] {{
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    z-index: 10;
}}
[class*="st-key-sc_card_"] [class*="st-key-sc_btn_"] .stButton {{
    width: 100%;
    height: 100%;
}}
[class*="st-key-sc_card_"] [class*="st-key-sc_btn_"] .stButton > button {{
    width: 100% !important;
    height: 100% !important;
    opacity: 0 !important;
    cursor: pointer !important;
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    transform: none !important;
}}
[class*="st-key-sc_card_"] [class*="st-key-sc_btn_"] .stButton > button:hover {{
    transform: none !important;
    box-shadow: none !important;
}}

/* Scenario card run button: raise above overlay + stretch to card height */
[class*="st-key-sc_card_"] [class*="st-key-sc_run_"] {{
    position: relative;
    z-index: 20;
}}
/* stLayoutWrapper wrapping the run container must stretch within the column */
[class*="st-key-sc_card_"] [data-testid="stLayoutWrapper"]:has([class*="st-key-sc_run_"]) {{
    flex: 1 !important;
}}
/* Propagate height through every intermediate wrapper */
[class*="st-key-sc_card_"] [class*="st-key-sc_run_"],
[class*="st-key-sc_card_"] [class*="st-key-sc_run_"] [data-testid="stElementContainer"],
[class*="st-key-sc_card_"] [class*="st-key-sc_run_"] [data-testid="stButton"],
[class*="st-key-sc_card_"] [class*="st-key-sc_run_"] .stButton > div,
[class*="st-key-sc_card_"] [class*="st-key-sc_run_"] [data-testid="stTooltipIcon"],
[class*="st-key-sc_card_"] [class*="st-key-sc_run_"] [data-testid="stTooltipHoverTarget"] {{
    flex: 1 !important;
    height: 100% !important;
}}
[class*="st-key-sc_card_"] [class*="st-key-sc_run_"] button {{
    height: 100% !important;
}}

/* Scenario card delete button: raise above overlay + stretch to card height */
[class*="st-key-sc_card_"] [class*="st-key-sc_del_"] {{
    position: relative;
    z-index: 20;
}}
[class*="st-key-sc_card_"] [data-testid="stLayoutWrapper"]:has([class*="st-key-sc_del_"]) {{
    flex: 1 !important;
}}
[class*="st-key-sc_card_"] [class*="st-key-sc_del_"],
[class*="st-key-sc_card_"] [class*="st-key-sc_del_"] [data-testid="stElementContainer"],
[class*="st-key-sc_card_"] [class*="st-key-sc_del_"] [data-testid="stButton"],
[class*="st-key-sc_card_"] [class*="st-key-sc_del_"] .stButton > div,
[class*="st-key-sc_card_"] [class*="st-key-sc_del_"] [data-testid="stTooltipIcon"],
[class*="st-key-sc_card_"] [class*="st-key-sc_del_"] [data-testid="stTooltipHoverTarget"] {{
    flex: 1 !important;
    height: 100% !important;
}}
[class*="st-key-sc_card_"] [class*="st-key-sc_del_"] button {{
    height: 100% !important;
}}

/* ── Action bar buttons ──────────────────── */
/* Specificity must exceed .stMainBlockContainer .stButton > button[kind] (0,3,1) */
.stMainBlockContainer .st-key-action_add_scenario .stButton button {{
    background: var(--teal-600) !important;
    color: var(--white) !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-xs) !important;
}}
.stMainBlockContainer .st-key-action_add_scenario .stButton button:hover {{
    background: var(--teal-700) !important;
    color: var(--white) !important;
    box-shadow: var(--shadow-sm) !important;
}}
.stMainBlockContainer .st-key-action_add_scenario button p,
.stMainBlockContainer .st-key-action_add_scenario button span {{
    color: var(--white) !important;
}}
.stMainBlockContainer .st-key-action_add_scenario {{
    padding-top: 1.5rem;
}}

/* Scenario run button: icon-only, compact, matching delete size */
.stMainBlockContainer [class*="st-key-sc_run_"] .stButton button {{
    padding: 0.375rem 0.5rem !important;
    min-width: 0 !important;
    font-size: 0 !important;
}}
.stMainBlockContainer [class*="st-key-sc_run_"] .stButton button span[data-testid="stIconMaterial"] {{
    font-size: 1.25rem !important;
}}

/* Scenario delete button: icon-only, transparent bg, red on hover */
.stMainBlockContainer [class*="st-key-sc_del_"] .stButton button {{
    background: transparent !important;
    color: var(--gray-500) !important;
    border: 1px solid var(--gray-300) !important;
    box-shadow: none !important;
    font-size: 0 !important;
    padding: 0.375rem 0.5rem !important;
    min-width: 0 !important;
}}
.stMainBlockContainer [class*="st-key-sc_del_"] .stButton button span[data-testid="stIconMaterial"] {{
    font-size: 1.25rem !important;
}}
.stMainBlockContainer [class*="st-key-sc_del_"] .stButton button:hover {{
    background: {COLORS['error_bg']} !important;
    color: var(--error) !important;
    border-color: var(--error) !important;
    box-shadow: none !important;
    transform: none !important;
}}
.stMainBlockContainer [class*="st-key-sc_del_"] button p,
.stMainBlockContainer [class*="st-key-sc_del_"] button span {{
    color: inherit !important;
}}

/* Scenario edit button: icon-only, transparent bg, teal on hover */
.stMainBlockContainer [class*="st-key-sc_edit_"] .stButton button {{
    background: transparent !important;
    color: var(--gray-500) !important;
    border: 1px solid var(--gray-300) !important;
    box-shadow: none !important;
    font-size: 0 !important;
    padding: 0.375rem 0.5rem !important;
    min-width: 0 !important;
}}
.stMainBlockContainer [class*="st-key-sc_edit_"] .stButton button span[data-testid="stIconMaterial"] {{
    font-size: 1.25rem !important;
}}
.stMainBlockContainer [class*="st-key-sc_edit_"] .stButton button:hover {{
    background: var(--teal-50) !important;
    color: var(--teal-600) !important;
    border-color: var(--teal-400) !important;
    box-shadow: none !important;
    transform: none !important;
}}
.stMainBlockContainer [class*="st-key-sc_edit_"] button p,
.stMainBlockContainer [class*="st-key-sc_edit_"] button span {{
    color: inherit !important;
}}

/* Scenario edit button: raise above overlay + stretch to card height */
[class*="st-key-sc_card_"] [class*="st-key-sc_edit_"] {{
    position: relative;
    z-index: 20;
}}
[class*="st-key-sc_card_"] [data-testid="stLayoutWrapper"]:has([class*="st-key-sc_edit_"]) {{
    flex: 1 !important;
}}
[class*="st-key-sc_card_"] [class*="st-key-sc_edit_"],
[class*="st-key-sc_card_"] [class*="st-key-sc_edit_"] [data-testid="stElementContainer"],
[class*="st-key-sc_card_"] [class*="st-key-sc_edit_"] [data-testid="stButton"],
[class*="st-key-sc_card_"] [class*="st-key-sc_edit_"] .stButton > div,
[class*="st-key-sc_card_"] [class*="st-key-sc_edit_"] [data-testid="stTooltipIcon"],
[class*="st-key-sc_card_"] [class*="st-key-sc_edit_"] [data-testid="stTooltipHoverTarget"] {{
    flex: 1 !important;
    height: 100% !important;
}}
[class*="st-key-sc_card_"] [class*="st-key-sc_edit_"] button {{
    height: 100% !important;
}}

/* AI analysis button (in title row) */
.stMainBlockContainer .st-key-action_ai .stButton button {{
    background: var(--teal-50) !important;
    border: 1px solid var(--teal-300) !important;
    color: var(--teal-600) !important;
    padding: 0.5rem 0.75rem !important;
    font-weight: 600 !important;
    font-size: 0.813rem !important;
    box-shadow: none !important;
    margin-top: 0.5rem !important;
}}
.stMainBlockContainer .st-key-action_ai button p,
.stMainBlockContainer .st-key-action_ai button span {{
    color: var(--teal-600) !important;
}}
.stMainBlockContainer .st-key-action_ai .stButton button:hover {{
    background: var(--teal-100) !important;
    border-color: var(--teal-400) !important;
    box-shadow: none !important;
}}
.stMainBlockContainer .st-key-action_ai .stButton button:disabled {{
    background: var(--gray-100) !important;
    border: none !important;
    color: var(--gray-400) !important;
    opacity: 0.6 !important;
    box-shadow: none !important;
}}
.stMainBlockContainer .st-key-action_ai button:disabled p,
.stMainBlockContainer .st-key-action_ai button:disabled span {{
    color: var(--gray-400) !important;
}}

/* Export popover trigger alignment in modal */
[data-testid="stDialog"] [class*="st-key-modal_action_export"] [data-testid="stPopover"] button {{
    margin-top: 0.5rem !important;
}}

/* Run button in modal */
[data-testid="stDialog"] [class*="st-key-modal_action_run"] .stButton button {{
    margin-top: 0.5rem !important;
}}
[data-testid="stDialog"] [class*="st-key-modal_action_run"] .stButton button:disabled {{
    background: var(--gray-200) !important;
    color: var(--gray-400) !important;
    border: none !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
    opacity: 0.7 !important;
}}
[data-testid="stDialog"] [class*="st-key-modal_action_run"] .stButton button:disabled p,
[data-testid="stDialog"] [class*="st-key-modal_action_run"] .stButton button:disabled span {{
    color: var(--gray-400) !important;
}}

/* Modal AI analysis button */
[data-testid="stDialog"] [class*="st-key-modal_action_ai"] .stButton button {{
    background: var(--teal-50) !important;
    border: 1px solid var(--teal-300) !important;
    color: var(--teal-600) !important;
    font-weight: 600 !important;
    font-size: 0.813rem !important;
    box-shadow: none !important;
    padding: 0.5rem 0.75rem !important;
    margin-top: 0.5rem !important;
}}
[data-testid="stDialog"] [class*="st-key-modal_action_ai"] button p,
[data-testid="stDialog"] [class*="st-key-modal_action_ai"] button span {{
    color: var(--teal-600) !important;
}}
[data-testid="stDialog"] [class*="st-key-modal_action_ai"] .stButton button:hover {{
    background: var(--teal-100) !important;
    border-color: var(--teal-400) !important;
    box-shadow: none !important;
}}
[data-testid="stDialog"] [class*="st-key-modal_action_ai"] .stButton button:disabled {{
    background: var(--gray-100) !important;
    border: none !important;
    color: var(--gray-400) !important;
    opacity: 0.6 !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
}}
[data-testid="stDialog"] [class*="st-key-modal_action_ai"] .stButton button:disabled p,
[data-testid="stDialog"] [class*="st-key-modal_action_ai"] .stButton button:disabled span {{
    color: var(--gray-400) !important;
}}

/* ── Experiment detail back button ───────── */
.stMainBlockContainer .st-key-detail_back .stButton button {{
    background: transparent !important;
    border: 1px solid var(--gray-300) !important;
    color: var(--gray-600) !important;
    padding: 0.375rem 0.75rem !important;
    font-size: 0.813rem !important;
    box-shadow: none !important;
}}
.stMainBlockContainer .st-key-detail_back button p,
.stMainBlockContainer .st-key-detail_back button span {{
    color: var(--gray-600) !important;
}}
.stMainBlockContainer .st-key-detail_back .stButton button:hover {{
    background: var(--gray-50) !important;
    border-color: var(--gray-400) !important;
    color: var(--gray-800) !important;
    box-shadow: none !important;
    transform: none !important;
}}

/* ── Parameters card ─────────────────────── */
.params-card {{
    background: {COLORS['white']};
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid {COLORS['gray_200']};
    height: 100%;
    min-height: 0;
}}
.params-card-title {{
    font-family: {FONTS['body']};
    font-size: 0.688rem;
    font-weight: 600;
    color: {COLORS['gray_500']};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.params-card-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.375rem 0;
    border-bottom: 1px solid {COLORS['gray_100']};
}}
.params-card-row:last-child {{
    border-bottom: none;
}}
.params-card-key {{
    font-family: {FONTS['body']};
    font-size: 0.813rem;
    color: {COLORS['gray_500']};
    font-weight: 500;
}}
.params-card-key-override {{
    font-family: {FONTS['body']};
    font-size: 0.813rem;
    color: {COLORS['teal_600']};
    font-weight: 600;
}}
.params-card-val {{
    font-family: {FONTS['mono']};
    font-size: 0.813rem;
    color: {COLORS['gray_900']};
    font-weight: 500;
}}
.params-card-val-override {{
    font-family: {FONTS['mono']};
    font-size: 0.813rem;
    color: {COLORS['teal_600']};
    font-weight: 600;
}}

/* ── Params section: equalize card heights ─ */
/* stLayoutWrapper wrapping the columns must stretch */
.st-key-params_section [data-testid="stLayoutWrapper"] {{
    flex: 1 !important;
}}
/* stElementContainer inside each column must fill the flex parent */
.st-key-params_section [data-testid="stColumn"] [data-testid="stElementContainer"] {{
    flex: 1 !important;
}}
/* Propagate height through Markdown wrappers to the card */
.st-key-params_section [data-testid="stMarkdown"] {{
    height: 100% !important;
}}
.st-key-params_section [data-testid="stMarkdown"] > div {{
    height: 100% !important;
    align-items: stretch !important;
}}
.st-key-params_section [data-testid="stMarkdownContainer"] {{
    height: 100% !important;
}}

/* ── Modal params section: equalize card heights ─ */
[class*="st-key-modal_params_section"] [data-testid="stLayoutWrapper"] {{
    flex: 1 !important;
}}
[class*="st-key-modal_params_section"] [data-testid="stColumn"] [data-testid="stElementContainer"] {{
    flex: 1 !important;
}}
[class*="st-key-modal_params_section"] [data-testid="stMarkdown"] {{
    height: 100% !important;
}}
[class*="st-key-modal_params_section"] [data-testid="stMarkdown"] > div {{
    height: 100% !important;
    align-items: stretch !important;
}}
[class*="st-key-modal_params_section"] [data-testid="stMarkdownContainer"] {{
    height: 100% !important;
}}

/* ── Spin animation for AI analyzing state ── */
@keyframes spin {{
    from {{ transform: rotate(0deg); }}
    to {{ transform: rotate(360deg); }}
}}

/* Spin the icon on AI buttons when disabled (analyzing state) */
.stMainBlockContainer .st-key-action_ai .stButton button:disabled
    span[data-testid="stIconMaterial"] {{
    animation: spin 1.5s linear infinite;
    display: inline-block;
}}

[data-testid="stDialog"] [class*="st-key-modal_action_ai"] .stButton button:disabled
    span[data-testid="stIconMaterial"] {{
    animation: spin 1.5s linear infinite;
    display: inline-block;
}}

/* ── AI analysis card ────────────────────── */
.ai-analysis-card {{
    background: {COLORS['white']};
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid {COLORS['gray_200']};
    margin-top: 1rem;
}}

/* ── Preferences page ──────────────────── */
[class*="st-key-pref_card"] {{
    background: var(--white);
    border-radius: var(--radius-lg);
    padding: 1rem 1.5rem 1.25rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--gray-200);
    margin-bottom: 0.5rem;
}}
.st-key-pref_card_danger {{
    border-color: var(--error) !important;
    border-style: dashed !important;
    background: #FFFBFB !important;
}}
.pref-sublabel {{
    font-family: var(--font-body);
    font-size: 0.688rem;
    font-weight: 600;
    color: var(--gray-500);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.375rem;
}}
.pref-sublabel svg {{
    color: var(--teal-500);
}}
/* Save preferences button */
.st-key-pref_save {{
    margin-top: 1rem;
}}
/* Danger-zone reset button */
.stMainBlockContainer .st-key-pref_reset .stButton button {{
    background: transparent !important;
    color: var(--error) !important;
    border: 1px solid var(--error) !important;
    box-shadow: none !important;
}}
.stMainBlockContainer .st-key-pref_reset .stButton button:hover {{
    background: var(--error) !important;
    color: var(--white) !important;
    box-shadow: none !important;
}}
.stMainBlockContainer .st-key-pref_reset button p,
.stMainBlockContainer .st-key-pref_reset button span {{
    color: inherit !important;
}}
/* Visible input borders inside preference cards */
[class*="st-key-pref_card"] input[type="text"],
[class*="st-key-pref_card"] input[type="number"],
[class*="st-key-pref_card"] input[type="password"] {{
    border: 1px solid var(--gray-300) !important;
    background: var(--gray-50) !important;
}}
/* Selectbox borders in pref cards */
[class*="st-key-pref_card"] .stSelectbox [data-baseweb="select"] > div {{
    border: 1px solid var(--gray-300) !important;
    border-radius: var(--radius-md) !important;
    background: var(--gray-50) !important;
}}

/* Center color pickers (label + swatch) in pref cards */
[class*="st-key-pref_card"] .stColorPicker {{
    display: flex;
    flex-direction: column;
    align-items: center;
}}
[class*="st-key-pref_card"] .stColorPicker [data-testid="stWidgetLabel"] {{
    width: 100%;
    justify-content: center;
}}
[class*="st-key-pref_card"] .stColorPicker [data-testid="stWidgetLabel"] label {{
    width: 100%;
    text-align: center;
}}
</style>
"""
    )


def stat_card(label: str, value: str, icon_svg: str = "") -> str:
    """Create a clean stat card with minimal teal accent."""
    icon_html = ""
    if icon_svg:
        icon_html = (
            f'<div style="width:36px;height:36px;background:{COLORS["teal_100"]};'
            f"border-radius:8px;display:flex;align-items:center;"
            f'justify-content:center;color:{COLORS["teal_500"]};flex-shrink:0;">'
            f"{icon_svg}</div>"
        )

    return _dedent(
        f"""
<div style="background:{COLORS['white']};border-radius:12px;padding:1.25rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,0.06),0 1px 2px -1px rgba(0,0,0,0.06);border:1px solid {COLORS['gray_200']};">
<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.875rem;">
{icon_html}
<div style="font-family:{FONTS['body']};font-size:0.688rem;font-weight:600;color:{COLORS['gray_500']};text-transform:uppercase;letter-spacing:0.06em;">{label}</div>
</div>
<div style="font-family:{FONTS['body']};font-size:1.75rem;font-weight:800;color:{COLORS['gray_900']};line-height:1;letter-spacing:-0.02em;">{value}</div>
</div>
"""
    )


def experiment_card(
    name: str,
    description: str,
    scenarios_complete: int,
    scenarios_total: int,
    last_run: str,
    status: str = "pending",
) -> str:
    """Create a clean experiment card with subtle hover-ready styling."""
    status_colors = {
        "pending": COLORS["gray_600"],
        "running": COLORS["info"],
        "complete": COLORS["success"],
        "failed": COLORS["error"],
    }

    status_bg = {
        "pending": COLORS["gray_100"],
        "running": COLORS["info_bg"],
        "complete": COLORS["success_bg"],
        "failed": COLORS["error_bg"],
    }

    progress = (scenarios_complete / scenarios_total * 100) if scenarios_total > 0 else 0

    return _dedent(
        f"""
<div class="exp-card" style="background:{COLORS['white']};border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:0.5rem;box-shadow:0 1px 3px rgba(0,0,0,0.06);border:1px solid {COLORS['gray_200']};transition:all 0.2s ease;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.625rem;">
<div style="font-family:{FONTS['body']};font-size:1rem;font-weight:700;color:{COLORS['gray_900']};letter-spacing:-0.01em;">{name}</div>
<div style="display:inline-block;padding:0.188rem 0.625rem;border-radius:99px;background:{status_bg.get(status, COLORS['gray_100'])};color:{status_colors.get(status, COLORS['gray_600'])};font-family:{FONTS['body']};font-size:0.688rem;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">{status}</div>
</div>
<div style="font-family:{FONTS['body']};font-size:0.813rem;color:{COLORS['gray_500']};margin-bottom:0.875rem;line-height:1.5;">{description}</div>
<div style="width:100%;height:4px;background:{COLORS['gray_200']};border-radius:2px;overflow:hidden;margin-bottom:0.875rem;">
<div style="height:100%;width:{progress}%;background:{COLORS['teal_500']};border-radius:2px;transition:width 0.3s ease;"></div>
</div>
<div style="display:flex;justify-content:space-between;align-items:center;font-family:{FONTS['mono']};font-size:0.75rem;color:{COLORS['gray_400']};">
<div>{scenarios_complete}/{scenarios_total} scenarios</div>
<div>{last_run}</div>
</div>
</div>
"""
    )


def page_header(title: str, subtitle: str = "") -> str:
    """Create a clean page header with proper hierarchy."""
    sub = ""
    if subtitle:
        sub = (
            f'<p style="font-family:{FONTS["body"]};font-size:0.938rem;'
            f'color:{COLORS["gray_500"]};margin-top:0.375rem;font-weight:400;'
            f'line-height:1.5;">{subtitle}</p>'
        )

    return _dedent(
        f"""
<div style="margin-bottom:2rem;">
<h1 style="font-family:{FONTS['body']};font-size:1.875rem;font-weight:800;color:{COLORS['gray_900']};margin:0;letter-spacing:-0.02em;">{title}</h1>
{sub}
</div>
"""
    )


def empty_state(title: str, message: str) -> str:
    """Create a clean empty state with centered content."""
    return _dedent(
        f"""
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:4rem 2rem;background:{COLORS['white']};border-radius:16px;box-shadow:0 1px 3px rgba(0,0,0,0.06);border:1px solid {COLORS['gray_200']};">
<div style="width:56px;height:56px;background:{COLORS['teal_100']};border-radius:14px;display:flex;align-items:center;justify-content:center;color:{COLORS['teal_500']};margin-bottom:1.5rem;">
<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6M9 3v9l-5 9h16l-5-9V3"/></svg>
</div>
<h2 style="font-family:{FONTS['body']};font-size:1.25rem;font-weight:700;color:{COLORS['gray_900']};margin-bottom:0.5rem;">{title}</h2>
<p style="font-family:{FONTS['body']};font-size:0.875rem;color:{COLORS['gray_500']};max-width:24rem;margin-bottom:0;line-height:1.6;">{message}</p>
</div>
"""
    )


def scenario_card(
    label: str,
    override_text: str,
    status: str = "created",
    progress: float = -1.0,
) -> str:
    """Create a scenario card with label, override summary, and status badge.

    Args:
        label: Scenario display name
        override_text: Summary of overridden parameters
        status: One of 'created', 'pending', 'running', 'completed', 'failed'
        progress: Progress fraction 0.0-1.0 when running, -1.0 for no bar
    """
    status_colors = {
        "created": (COLORS["teal_500"], COLORS["teal_100"]),
        "edited": (COLORS["teal_500"], COLORS["teal_100"]),
        "pending": (COLORS["gray_600"], COLORS["gray_100"]),
        "running": (COLORS["info"], COLORS["info_bg"]),
        "completed": (COLORS["success"], COLORS["success_bg"]),
        "failed": (COLORS["error"], COLORS["error_bg"]),
    }

    s_color, s_bg = status_colors.get(status, (COLORS["gray_600"], COLORS["gray_100"]))
    s_text = status.upper()

    # Animated pulsing dot for running status
    badge_prefix = ""
    if status == "running":
        badge_prefix = (
            '<span style="display:inline-block;width:6px;height:6px;'
            f'border-radius:50%;background:{COLORS["info"]};'
            "margin-right:0.375rem;vertical-align:middle;"
            'animation:pulse-dot 1.5s ease-in-out infinite;"></span>'
        )

    override_html = ""
    if override_text:
        override_html = (
            f'<div style="font-family:{FONTS["body"]};font-size:0.75rem;'
            f'color:{COLORS["gray_500"]};margin-top:0.375rem;'
            f'line-height:1.5;">{override_text}</div>'
        )
    else:
        override_html = (
            f'<div style="font-family:{FONTS["body"]};font-size:0.75rem;'
            f'color:{COLORS["gray_400"]};margin-top:0.375rem;'
            f'font-style:italic;">Using all global defaults</div>'
        )

    # Inline progress bar for running simulations
    progress_html = ""
    if status == "running" and progress >= 0:
        pct = max(0.0, min(1.0, progress)) * 100
        pct_text = f"{pct:.0f}%"
        progress_html = (
            f'<div style="flex:1;display:flex;align-items:center;margin:0 1rem;">'
            '<div class="scenario-progress-bar" style="flex:1;">'
            f'<div class="scenario-progress-fill" style="width:{pct}%;"></div>'
            "</div>"
            f'<span style="font-family:{FONTS["body"]};font-size:0.688rem;'
            f'color:{COLORS["gray_500"]};margin-left:0.5rem;'
            f'white-space:nowrap;">{pct_text}</span>'
            "</div>"
        )

    badge_html = (
        f'<div style="display:inline-flex;align-items:center;'
        f"padding:0.188rem 0.625rem;border-radius:99px;background:{s_bg};"
        f'color:{s_color};font-family:{FONTS["body"]};font-size:0.625rem;'
        f"font-weight:600;text-transform:uppercase;letter-spacing:0.04em;"
        f'flex-shrink:0;">{badge_prefix}{s_text}</div>'
    )

    title_html = (
        f'<div style="font-family:{FONTS["body"]};font-size:0.938rem;'
        f'font-weight:600;color:{COLORS["gray_900"]};letter-spacing:-0.01em;'
        f'flex-shrink:0;">{label}</div>'
    )

    return _dedent(
        f"""
<div class="scenario-card" style="background:{COLORS['white']};border-radius:12px;padding:1rem 1.25rem;box-shadow:0 1px 3px rgba(0,0,0,0.06);border:1px solid {COLORS['gray_200']};transition:all 0.2s ease;margin-bottom:0.375rem;">
<div style="display:flex;justify-content:space-between;align-items:center;">
{title_html}
{progress_html}
{badge_html}
</div>
{override_html}
</div>
"""
    )


def params_card(title: str, icon_svg: str, rows: list) -> str:
    """Create a parameter display card with key-value rows.

    Args:
        title: Card title (e.g. "Network", "Distribution")
        icon_svg: SVG icon HTML string
        rows: List of (key, value) or (key, value, is_override) tuples
    """
    rows_html = ""
    for row in rows:
        key, val = row[0], row[1]
        is_override = row[2] if len(row) > 2 else False
        key_class = "params-card-key-override" if is_override else "params-card-key"
        val_class = "params-card-val-override" if is_override else "params-card-val"
        rows_html += (
            f'<div class="params-card-row">'
            f'<span class="{key_class}">{key}</span>'
            f'<span class="{val_class}">{val}</span>'
            f"</div>"
        )

    icon_html = ""
    if icon_svg:
        icon_html = (
            f'<span style="color:{COLORS["teal_500"]};display:flex;'
            f'align-items:center;">{icon_svg}</span>'
        )

    return _dedent(
        f"""
<div class="params-card">
<div class="params-card-title">{icon_html}{title}</div>
{rows_html}
</div>
"""
    )


def circular_progress_html(progress: float, label: str = "Running simulation...") -> str:
    """Create a CSS-only circular progress ring.

    Args:
        progress: Fraction between 0.0 and 1.0
        label: Text shown below the ring
    """
    pct = max(0.0, min(1.0, progress))
    deg = int(pct * 360)
    pct_text = f"{int(pct * 100)}%"

    return _dedent(
        f"""
<div style="display:flex;flex-direction:column;align-items:center;
justify-content:center;min-height:450px;padding:2rem 1rem;">
<div style="width:360px;height:360px;border-radius:50%;
background:conic-gradient({COLORS['teal_500']} {deg}deg, {COLORS['gray_200']} {deg}deg);
display:flex;align-items:center;justify-content:center;">
<div style="width:282px;height:282px;border-radius:50%;background:{COLORS['white']};
display:flex;align-items:center;justify-content:center;
font-family:{FONTS['body']};font-size:3rem;font-weight:700;
color:{COLORS['gray_800']};">{pct_text}</div>
</div>
<p style="font-family:{FONTS['body']};font-size:0.938rem;
color:{COLORS['gray_500']};margin-top:1.25rem;font-weight:500;">{label}</p>
</div>
"""
    )


def section_header(title: str, subtitle: str = "") -> str:
    """Create a section header for content areas."""
    sub = ""
    if subtitle:
        sub = (
            f'<p style="font-family:{FONTS["body"]};font-size:0.813rem;'
            f'color:{COLORS["gray_500"]};margin-top:0.25rem;">{subtitle}</p>'
        )

    return _dedent(
        f"""
<div style="margin:1.5rem 0 1rem 0;">
<h2 style="font-family:{FONTS['body']};font-size:1.25rem;font-weight:700;color:{COLORS['gray_800']};margin:0;letter-spacing:-0.01em;">{title}</h2>
{sub}
</div>
"""
    )
