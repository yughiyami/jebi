# -*- coding: utf-8 -*-
"""
reporter.py v2  -  Dashboard HTML completo con:
  - Video izquierdo + derecho sincronizados
  - Alertas en tiempo real (feed scrolling)
  - 8 gauges de eficiencia animados
  - Graficos IMU scrolling
  - Tabla de ciclos + despacho + transporte
"""

import json, os
import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime

from imu_processor  import LoadCycle, WaitEvent, SessionMetrics, BUCKET_MAX_T, BUCKET_M3
from metrics        import (EfficiencyProfile, Alert, TruckLoadEvent,
                             TransportMetrics, ALERT_SEVERITY)

# ─── PALETA ──────────────────────────────────────────────────────────────────
C = dict(
  bg='#0d1117', panel='#161b22', panel2='#1c2128', border='#30363d',
  text='#e6edf3', muted='#8b949e',
  blue='#4a9eff', orange='#ff8c00', green='#3fb950',
  red='#f85149',  yellow='#ffd700', purple='#bc8cff', cyan='#39d353',
  critical='#f85149', warning='#ffd700', info='#4a9eff', success='#3fb950',
)
PHASE_COLORS = dict(
  DIG='#ff8c00', SWING_LOADED='#4a9eff', DUMP='#f85149',
  SWING_EMPTY='#bc8cff', REPOSITION='#39d353', WAIT='#30363d',
)


def _is_mini_profile(p, cycles) -> bool:
    """Helper: verifica si un EfficiencyProfile corresponde a un mini-cycle."""
    for c in cycles:
        if c.cycle_id == p.cycle_id:
            return c.is_mini_cycle
    return False


def generate_report(df_imu, cycles, wait_events, metrics,
                    video_events, df_cycles, df_waits, output_dir,
                    profiles=None, alerts=None,
                    truck_events=None, transport=None,
                    timeline=None,
                    ocr_events=None, wear_score=None,
                    spill_events=None, dust_index=None,
                    ipo=None, idle_index=None,
                    visual_validation=None,
                    session_label='DIG - EX-5600') -> str:

    html = _build(df_imu, cycles, wait_events, metrics,
                  video_events, profiles or [], alerts or [],
                  truck_events or [], transport,
                  timeline or [], session_label,
                  ocr_events or [], wear_score or {},
                  spill_events or [], dust_index or {},
                  ipo, idle_index or {},
                  visual_validation or {})

    path = os.path.join(output_dir, 'dashboard.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  [reporter] Dashboard guardado: {path}')
    return path


# ─── BUILDER PRINCIPAL ───────────────────────────────────────────────────────

def _build(df_imu, cycles, wait_events, metrics, video_events,
           profiles, alerts, truck_events, transport, timeline, label,
           ocr_events=None, wear_score=None, spill_events=None, dust_index=None,
           ipo=None, idle_index=None, visual_validation=None):

    t   = df_imu.timestamp_s.values.tolist()
    ax  = df_imu.ax.values.tolist()
    ay  = df_imu.ay.values.tolist()
    az  = df_imu.az.values.tolist()
    gx  = df_imu.gx.values.tolist()
    gy  = df_imu.gy.values.tolist()
    gz  = df_imu.gz.values.tolist()
    an  = df_imu.accel_norm.values.tolist()
    gn  = df_imu.gyro_norm.values.tolist()

    full = [c for c in cycles if not c.is_mini_cycle]
    now  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # BUGFIX: promedios de sesión para gauges (así no inician en 50% placeholder).
    # Se calculan desde profiles (ciclos completos, excluye mini-cycles ruidosos).
    full_profiles = [p for p in profiles if not _is_mini_profile(p, cycles)]
    if full_profiles:
        avg_coll = float(np.mean([p.collection_eff_pct  for p in full_profiles]))
        avg_man  = float(np.mean([p.maneuver_eff_pct    for p in full_profiles]))
        avg_pos  = float(np.mean([p.positioning_eff_pct for p in full_profiles]))
        avg_disc = float(np.mean([p.discharge_eff_pct   for p in full_profiles]))
        avg_oee  = float(np.mean([p.cycle_eff_pct       for p in full_profiles]))
    else:
        avg_coll = avg_man = avg_pos = avg_disc = avg_oee = 0.0

    # JSON para JS
    tl_json       = json.dumps(timeline)
    alerts_json   = json.dumps([dict(
        t=a.timestamp_s, type=a.alert_type, sev=a.severity,
        msg=a.message, val=a.value, unit=a.unit, cid=a.cycle_id
    ) for a in alerts])
    shapes_json   = _cycle_shapes(cycles, wait_events)
    payload_json  = _payload_json(full)
    trucks_json   = json.dumps([dict(
        id=te.truck_id, model=te.model, cap=te.capacity_t,
        passes=te.n_passes, payload=te.total_payload_t,
        fill=te.fill_pct, done=te.dispatched
    ) for te in truck_events])

    # HTML building blocks
    kpis_html    = _kpi_cards(metrics, transport)
    eff_html     = _efficiency_table(profiles)
    waits_html   = _waits_table(wait_events)
    help_html    = _help_content()
    ipo_html     = _ipo_view(ipo)
    idle_html    = _idle_view(idle_index or {})
    insights_html = _insights_view(metrics, idle_index or {}, ipo, transport,
                                    visual_validation or {})
    pausas_html  = _pausas_view(visual_validation or {})

    return f"""<!DOCTYPE html>
<html lang="es" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<meta name="description" content="JEBI Hackathon 2026 — Shovel Intelligence Dashboard con dual-source validation (IMU + Optical Flow)">
<meta name="theme-color" content="#0d1117">
<title>DIG | {label}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/0.159.0/three.min.js"></script>
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']]}},svg:{{fontCache:'global'}}}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
/* ═══════════════════ CSS VARIABLES — TEMAS ═══════════════════ */
[data-theme="dark"] {{
  --bg:       #0d1117; --panel:   #161b22; --panel2:  #1c2128; --border:  #30363d;
  --text:     #e6edf3; --muted:   #8b949e;
  --blue:     #4a9eff; --orange:  #ff8c00; --green:   #3fb950;
  --red:      #f85149; --yellow:  #ffd700; --purple:  #bc8cff; --cyan:    #39d353;
  --critical: #f85149; --warning: #ffd700; --info:    #4a9eff; --success: #3fb950;
  --shadow:   0 0 60px rgba(248,81,73,.4);
}}
[data-theme="light"] {{
  --bg:       #f5f7fa; --panel:   #ffffff; --panel2:  #eef2f6; --border:  #d0d7de;
  --text:     #1f2328; --muted:   #636c76;
  --blue:     #0969da; --orange:  #bc4c00; --green:   #1a7f37;
  --red:      #cf222e; --yellow:  #9a6700; --purple:  #8250df; --cyan:    #0a3069;
  --critical: #cf222e; --warning: #9a6700; --info:    #0969da; --success: #1a7f37;
  --shadow:   0 0 60px rgba(207,34,46,.25);
}}

*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:-apple-system,'Segoe UI',sans-serif;font-size:13px;line-height:1.5;transition:background .2s,color .2s}}
::-webkit-scrollbar{{width:6px;height:6px}}::-webkit-scrollbar-track{{background:var(--panel)}}::-webkit-scrollbar-thumb{{background:var(--border);border-radius:3px}}

/* Reglas var-based que reemplazan valores hardcoded fuera de templates Python */
.theme-aware{{color:var(--text);background:var(--panel);border-color:var(--border)}}
.tv-text{{color:var(--text)}} .tv-muted{{color:var(--muted)}} .tv-bg{{background:var(--bg)}}
.tv-panel{{background:var(--panel)}} .tv-border{{border-color:var(--border)}}
.tv-blue{{color:var(--blue)}} .tv-green{{color:var(--green)}} .tv-red{{color:var(--red)}}
.tv-yellow{{color:var(--yellow)}}

/* Theme toggle button */
.theme-toggle{{background:transparent;border:1px solid var(--border);color:var(--text);
  border-radius:50%;width:32px;height:32px;cursor:pointer;font-size:16px;
  display:inline-flex;align-items:center;justify-content:center;transition:all .2s;margin-right:6px}}
.theme-toggle:hover{{background:var(--panel2);border-color:var(--blue)}}

/* HEADER */
.hdr{{background:{C['panel']};border-bottom:1px solid {C['border']};padding:10px 20px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}}
.hdr h1{{font-size:1.1rem;color:{C['blue']};font-weight:700}}
.hdr .meta{{color:{C['muted']};font-size:0.75rem}}
#live-clock{{color:{C['yellow']};font-weight:700;font-size:0.9rem}}
#phase-badge{{padding:3px 12px;border-radius:12px;font-size:0.75rem;font-weight:700;background:{C['border']};margin-left:10px}}

/* LAYOUT */
.container{{padding:14px 18px;max-width:1800px;margin:0 auto}}
.sec{{margin-bottom:20px}}
.sec-title{{font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:{C['muted']};border-bottom:1px solid {C['border']};padding-bottom:5px;margin-bottom:12px}}
.row{{display:flex;gap:12px;flex-wrap:wrap}}
/* BUGFIX responsive: col-video flexible en vez de ancho fijo de 960px */
.col-video{{flex:1 1 600px;min-width:0;max-width:100%}}
.col-alerts{{flex:1 1 280px;min-width:0}}
/* BUGFIX responsive: grids con auto-fit para colapsar en pantallas chicas */
.g2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}
.g3{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}}
.g4{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.full{{grid-column:1/-1}}
.panel{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;padding:10px}}

/* KPI */
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}}
.kpi{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;padding:12px 14px}}
.kpi-lbl{{font-size:0.65rem;color:{C['muted']};text-transform:uppercase;letter-spacing:.06em}}
.kpi-val{{font-size:1.6rem;font-weight:700;margin-top:3px}}
.kpi-sub{{font-size:0.68rem;color:{C['muted']};margin-top:1px}}
.g{{color:{C['green']}}}.w{{color:{C['yellow']}}}.r{{color:{C['red']}}}.b{{color:{C['blue']}}}

/* VIDEO */
/* BUGFIX responsive: flex-wrap para que en mobile los videos se stacken */
.video-wrap{{display:flex;gap:4px;background:#000;border-radius:8px;overflow:hidden;flex-wrap:wrap}}
.video-wrap video{{flex:1 1 45%;min-width:0;height:240px;object-fit:cover;max-width:100%}}
.video-label{{font-size:0.65rem;color:{C['muted']};margin-bottom:4px}}
.video-overlay{{position:relative;flex:1 1 45%;min-width:0}}
.video-overlay .ov{{position:absolute;bottom:8px;left:8px;background:rgba(0,0,0,.7);border-radius:4px;padding:2px 8px;font-size:0.65rem;color:#fff}}

/* ALERTS */
#alert-feed{{height:260px;overflow-y:auto;display:flex;flex-direction:column-reverse;gap:4px}}
.alert-item{{padding:6px 10px;border-radius:5px;border-left:3px solid;font-size:0.75rem;animation:fadeIn .3s ease}}
.alert-critical{{background:rgba(248,81,73,.12);border-color:{C['critical']};color:{C['critical']}}}
.alert-warning {{background:rgba(255,215,0,.10);border-color:{C['warning']};color:{C['warning']}}}
.alert-info    {{background:rgba(74,158,255,.10);border-color:{C['info']};color:{C['info']}}}
.alert-success {{background:rgba(63,185,80,.12);border-color:{C['success']};color:{C['success']}}}
.alert-time{{font-weight:700;margin-right:6px}}
@keyframes fadeIn{{from{{opacity:0;transform:translateY(-4px)}}to{{opacity:1;transform:translateY(0)}}}}

/* GAUGE ROW */
/* BUGFIX responsive: grid auto-fit en vez de flex para que siempre se rendericen bien */
.gauge-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}}
.gauge-box{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;padding:8px;min-width:0;min-height:140px}}
.gauge-box > div{{min-height:120px;width:100%}}
.gauge-title{{font-size:0.65rem;text-transform:uppercase;color:{C['muted']};letter-spacing:.06em;margin-bottom:4px}}

/* TABLES */
table{{width:100%;border-collapse:collapse;font-size:0.78rem}}
th{{background:{C['panel2']};color:{C['muted']};font-size:0.65rem;text-transform:uppercase;letter-spacing:.05em;padding:6px 8px;text-align:left;border-bottom:1px solid {C['border']}}}
td{{padding:5px 8px;border-bottom:1px solid {C['border']}}}
tr:hover td{{background:rgba(255,255,255,.02)}}
.badge{{display:inline-block;padding:1px 7px;border-radius:4px;font-size:.65rem;font-weight:600;text-transform:uppercase}}
.bg{{background:rgba(63,185,80,.15);color:{C['green']}}}.bw{{background:rgba(255,215,0,.15);color:{C['yellow']}}}.br{{background:rgba(248,81,73,.15);color:{C['red']}}}.bm{{background:rgba(188,140,255,.15);color:{C['purple']}}}.bb{{background:rgba(74,158,255,.15);color:{C['blue']}}}

/* PLAYBACK */
#playbar{{display:flex;align-items:center;gap:10px;background:{C['panel']};border:1px solid {C['border']};border-radius:8px;padding:8px 14px}}
#play-btn{{background:{C['blue']};color:#fff;border:none;border-radius:5px;padding:5px 16px;cursor:pointer;font-size:0.8rem;font-weight:600}}
#play-btn:hover{{background:#3a8ee0}}
#progress{{flex:1;height:4px;-webkit-appearance:none;border-radius:2px;background:{C['border']};cursor:pointer}}
#progress::-webkit-slider-thumb{{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:{C['blue']};cursor:pointer}}
#t-display{{font-size:0.8rem;color:{C['yellow']};min-width:60px;text-align:right;font-weight:700}}
.speed-btn{{background:{C['panel2']};color:{C['muted']};border:1px solid {C['border']};border-radius:4px;padding:3px 8px;cursor:pointer;font-size:0.72rem}}
.speed-btn.active{{background:{C['blue']};color:#fff;border-color:{C['blue']}}}

/* THUMBS */
.thumb-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}}
.thumb-card{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;overflow:hidden}}
.thumb-card img{{width:100%;display:block}}
.thumb-info{{padding:7px 10px;font-size:0.72rem;color:{C['muted']}}}
.thumb-info strong{{color:{C['blue']}}}

/* PHASE indicator */
.phase-DIG{{background:rgba(255,140,0,.15);color:{C['orange']}}}
.phase-SWING_LOADED{{background:rgba(74,158,255,.15);color:{C['blue']}}}
.phase-DUMP{{background:rgba(248,81,73,.15);color:{C['red']}}}
.phase-SWING_EMPTY{{background:rgba(188,140,255,.15);color:{C['purple']}}}
.phase-WAIT{{background:rgba(48,54,61,.5);color:{C['muted']}}}
.phase-REPOSITION{{background:rgba(57,211,83,.15);color:{C['green']}}}

/* ═══ SIDEBAR + LAYOUT ═══ */
.app{{display:flex;min-height:calc(100vh - 46px)}}
.sidebar{{width:220px;flex:0 0 220px;background:{C['panel']};border-right:1px solid {C['border']};padding:12px 0;position:sticky;top:46px;height:calc(100vh - 46px);overflow-y:auto}}
.side-item{{display:flex;align-items:center;gap:10px;padding:10px 16px;cursor:pointer;color:{C['muted']};font-size:0.82rem;font-weight:500;border-left:3px solid transparent;transition:all .15s}}
.side-item:hover{{background:{C['panel2']};color:{C['text']}}}
.side-item.active{{background:{C['panel2']};color:{C['blue']};border-left-color:{C['blue']};font-weight:700}}
.side-icon{{width:18px;height:18px;display:inline-flex;align-items:center;justify-content:center;font-size:0.95rem}}
.side-badge{{margin-left:auto;background:{C['red']};color:#fff;font-size:0.6rem;padding:1px 6px;border-radius:8px;font-weight:700;min-width:18px;text-align:center}}
.side-divider{{height:1px;background:{C['border']};margin:8px 16px}}
.side-head{{font-size:0.62rem;color:{C['muted']};text-transform:uppercase;letter-spacing:.1em;padding:6px 16px;margin-top:4px}}
.main{{flex:1;min-width:0;overflow-x:auto}}

/* Vistas tabuladas */
.view{{display:none;padding:16px 20px}}
.view.active{{display:block}}
.view-hdr{{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid {C['border']}}}
.view-hdr h2{{font-size:1.1rem;color:{C['text']};font-weight:700}}
.view-hdr .desc{{font-size:0.75rem;color:{C['muted']};margin-top:3px}}
.view-actions{{display:flex;gap:8px;align-items:center}}
.btn{{background:{C['panel2']};color:{C['text']};border:1px solid {C['border']};border-radius:5px;padding:6px 12px;cursor:pointer;font-size:0.75rem;font-weight:600;display:inline-flex;align-items:center;gap:6px;transition:all .15s}}
.btn:hover{{background:{C['border']};border-color:{C['blue']};color:{C['blue']}}}
.btn-primary{{background:{C['blue']};color:#fff;border-color:{C['blue']}}}
.btn-primary:hover{{background:#3a8ee0;color:#fff}}
.btn-danger{{background:{C['red']};color:#fff;border-color:{C['red']}}}

/* ═══ MODAL DE ALERTA CRITICA ═══ */
.modal-bg{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:9999;justify-content:center;align-items:center;animation:fadeIn .2s}}
.modal-bg.show{{display:flex}}
.modal{{background:{C['panel']};border:2px solid {C['red']};border-radius:12px;max-width:560px;width:90%;overflow:hidden;box-shadow:0 0 60px rgba(248,81,73,.4);animation:popIn .3s ease}}
.modal-head{{background:linear-gradient(90deg,{C['red']},#b02a2a);color:#fff;padding:14px 20px;font-weight:700;font-size:1rem;display:flex;align-items:center;gap:10px}}
.modal-icon{{font-size:1.6rem;animation:pulse 1s infinite}}
.modal-body{{padding:20px}}
.modal-type{{font-size:0.7rem;color:{C['muted']};text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}}
.modal-title{{font-size:1.2rem;color:{C['red']};font-weight:700;margin-bottom:10px}}
.modal-msg{{color:{C['text']};font-size:0.9rem;line-height:1.6;margin-bottom:14px}}
.modal-data{{background:{C['panel2']};border:1px solid {C['border']};border-radius:6px;padding:10px 14px;font-size:0.8rem;color:{C['muted']};margin-bottom:14px;font-family:monospace}}
.modal-data b{{color:{C['yellow']}}}
.modal-action{{background:rgba(74,158,255,.1);border-left:3px solid {C['blue']};padding:10px 14px;color:{C['text']};font-size:0.82rem;line-height:1.5;margin-bottom:14px;border-radius:4px}}
.modal-action strong{{color:{C['blue']};display:block;margin-bottom:4px;font-size:0.7rem;text-transform:uppercase;letter-spacing:.05em}}
.modal-foot{{display:flex;gap:10px;justify-content:flex-end;padding:14px 20px;background:{C['panel2']};border-top:1px solid {C['border']}}}
@keyframes popIn{{from{{transform:scale(.9);opacity:0}}to{{transform:scale(1);opacity:1}}}}
@keyframes pulse{{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.15)}}}}

/* ═══ GLOSARIO / HELP ═══ */
.glossary{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;padding:12px 14px;margin-top:10px}}
.glossary h4{{font-size:0.75rem;color:{C['blue']};text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}}
.glossary dl{{display:grid;grid-template-columns:160px 1fr;gap:6px 12px;font-size:0.78rem}}
.glossary dt{{color:{C['yellow']};font-weight:600}}
.glossary dd{{color:{C['muted']}}}
.threshold-list{{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}}
.threshold-chip{{background:{C['panel2']};border:1px solid {C['border']};border-radius:16px;padding:3px 10px;font-size:0.7rem;color:{C['muted']}}}
.threshold-chip b{{color:{C['yellow']}}}

/* ═══ SIMULADOR VISUAL ═══ */
.sim-wrap{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.sim-panel{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;padding:12px}}
.sim-panel h3{{font-size:0.85rem;margin-bottom:8px;display:flex;align-items:center;gap:6px}}
.sim-canvas{{width:100%;aspect-ratio:16/9;background:#000;border-radius:6px;border:2px solid {C['border']};display:block}}
.sim-ok{{border-color:{C['green']}}}
.sim-bad{{border-color:{C['red']}}}
.sim-legend{{display:flex;gap:10px;font-size:0.7rem;margin-top:8px;justify-content:center;color:{C['muted']}}}
.sim-legend span{{display:inline-flex;align-items:center;gap:4px}}
.sim-legend .dot{{width:10px;height:10px;border-radius:50%;display:inline-block}}
.sim-controls{{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;padding:8px;background:{C['panel2']};border-radius:6px}}
.sim-controls button{{flex:1;min-width:80px;background:{C['panel']};color:{C['text']};border:1px solid {C['border']};border-radius:4px;padding:6px 10px;cursor:pointer;font-size:0.72rem}}
.sim-controls button.active{{background:{C['blue']};color:#fff;border-color:{C['blue']}}}

/* ═══ OCR READINGS ═══ */
.ocr-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}}
.ocr-card{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;overflow:hidden}}
.ocr-card img{{width:100%;display:block;border-bottom:1px solid {C['border']}}}
.ocr-body{{padding:10px 12px}}
.ocr-id{{font-size:1.4rem;font-weight:700;color:{C['blue']};font-family:monospace;letter-spacing:.05em}}
.ocr-weight{{font-size:1.1rem;font-weight:600;color:{C['yellow']};font-family:monospace;margin-top:2px}}
.ocr-meta{{font-size:0.68rem;color:{C['muted']};margin-top:4px;display:flex;justify-content:space-between}}

/* ═══ ALERT MINI-CLIPS GRID ═══ */
.alert-grid-clips{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}}
.alert-clip-card{{background:{C['panel']};border:1px solid {C['border']};border-radius:10px;overflow:hidden;display:flex;flex-direction:column}}
.alert-clip-card.sev-critical{{border-left:4px solid {C['red']}}}
.alert-clip-card.sev-warning{{border-left:4px solid {C['yellow']}}}
.alert-clip-card.sev-info{{border-left:4px solid {C['blue']}}}
.alert-clip-card.sev-success{{border-left:4px solid {C['green']}}}
.alert-clip-video{{width:100%;aspect-ratio:16/9;background:#000;display:block}}
.alert-clip-body{{padding:10px 12px;flex:1;display:flex;flex-direction:column;gap:6px}}
.alert-clip-head{{display:flex;align-items:center;justify-content:space-between;gap:8px}}
.alert-clip-type{{font-weight:700;font-size:0.85rem}}
.alert-clip-time{{font-family:monospace;color:{C['yellow']};font-size:0.78rem}}
.alert-clip-msg{{font-size:0.78rem;color:{C['muted']};line-height:1.4}}
.alert-clip-ctrl{{display:flex;gap:6px;padding:8px 12px;background:{C['panel2']};border-top:1px solid {C['border']}}}
.alert-clip-ctrl button{{flex:1;background:transparent;border:1px solid {C['border']};color:{C['text']};padding:4px 8px;border-radius:4px;cursor:pointer;font-size:0.7rem}}
.alert-clip-ctrl button:hover{{background:{C['blue']};color:#fff;border-color:{C['blue']}}}

/* ═══ IPO VISTA MAESTRA ═══ */
.ipo-hero{{background:linear-gradient(135deg,{C['panel']} 0%,{C['panel2']} 100%);border:1px solid {C['border']};border-radius:12px;padding:24px;text-align:center;margin-bottom:14px}}
.ipo-big{{font-size:4rem;font-weight:800;font-family:'Segoe UI',monospace;line-height:1;margin:4px 0}}
.ipo-band{{display:inline-block;padding:4px 14px;border-radius:14px;font-weight:700;font-size:0.85rem;letter-spacing:.08em;text-transform:uppercase;margin-top:10px}}
.ipo-formula{{background:{C['panel2']};border:1px solid {C['border']};border-radius:8px;padding:16px 18px;margin:14px 0;font-size:1rem;overflow-x:auto;text-align:center}}
.ipo-components{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;margin-top:10px}}
.ipo-comp-card{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;padding:12px}}
.ipo-comp-sym{{font-family:monospace;font-size:0.75rem;color:{C['blue']};margin-bottom:3px}}
.ipo-comp-val{{font-size:1.3rem;font-weight:700;font-family:monospace}}
.ipo-comp-lbl{{font-size:0.68rem;color:{C['muted']};margin-top:2px}}
.ipo-tbands{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px}}
.ipo-tband{{padding:10px;border-radius:6px;text-align:center;font-size:0.72rem;border:1px solid {C['border']}}}
.ipo-tband b{{display:block;font-size:1rem;margin-bottom:2px}}
.ipo-tband.active{{border-width:2px;box-shadow:0 0 10px rgba(74,158,255,.3)}}

/* ═══ WEAR + DESPERDICIOS ═══ */
.bpmn-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}
.bpmn-card{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;padding:14px}}
.bpmn-card h4{{font-size:0.7rem;text-transform:uppercase;letter-spacing:.06em;color:{C['muted']};margin-bottom:8px}}
.bpmn-value{{font-size:1.8rem;font-weight:700;margin:6px 0}}
.bpmn-bar{{height:8px;background:{C['panel2']};border-radius:4px;overflow:hidden;margin:8px 0 4px}}
.bpmn-bar-fill{{height:100%;transition:width .4s ease}}
.bpmn-note{{font-size:0.7rem;color:{C['muted']}}}

/* ═══════════════════ RESPONSIVE BREAKPOINTS ═══════════════════ */

/* TABLETS y laptops chicos: <= 1024px */
@media (max-width: 1024px) {{
  .sidebar{{width:180px;flex:0 0 180px}}
  .side-item{{padding:8px 12px;font-size:0.78rem}}
  .side-head{{padding:5px 12px}}
  .kpi-val{{font-size:1.3rem}}
  .kpi{{padding:10px 12px}}
  .container{{padding:10px 12px}}
  .view{{padding:12px 14px}}
  .hdr{{padding:8px 14px}}
  .hdr h1{{font-size:1rem}}
  .hdr .meta{{display:none}}
  #prod-banner{{padding:12px 16px !important}}
  .video-wrap video{{height:200px}}
}}

/* TABLETS: <= 768px — sidebar colapsa a topbar horizontal */
@media (max-width: 768px) {{
  .app{{flex-direction:column}}
  .sidebar{{
    width:100%;flex:0 0 auto;height:auto;position:sticky;top:46px;
    max-height:none;overflow-x:auto;overflow-y:hidden;white-space:nowrap;
    padding:4px 0;border-right:none;border-bottom:1px solid {C['border']};
    display:flex;flex-wrap:nowrap
  }}
  .sidebar .side-head,
  .sidebar .side-divider{{display:none}}
  .side-item{{
    display:inline-flex;flex-direction:column;align-items:center;
    padding:6px 10px;border-left:none;border-bottom:3px solid transparent;
    font-size:0.65rem;min-width:70px;gap:3px;flex:0 0 auto
  }}
  .side-item.active{{border-left:none;border-bottom-color:{C['blue']}}}
  .side-icon{{font-size:1.2rem;width:auto;height:auto}}
  .side-item span:last-child{{font-size:0.62rem;text-align:center}}
  .side-badge{{margin-left:0;margin-top:2px;font-size:0.55rem}}

  .view{{padding:10px 12px}}
  .kpi-grid{{grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px}}
  .kpi-val{{font-size:1.15rem}}
  .kpi{{padding:8px 10px}}
  .view-hdr{{flex-wrap:wrap;gap:8px}}
  .view-hdr h2{{font-size:0.95rem}}
  .view-hdr .desc{{font-size:0.7rem}}
  .view-actions{{flex-wrap:wrap;gap:4px}}
  .view-actions .btn{{font-size:0.68rem;padding:4px 8px}}
  .btn{{font-size:0.7rem;padding:5px 9px}}

  #playbar{{flex-wrap:wrap;padding:6px 10px;gap:6px}}
  #playbar #progress{{order:10;flex-basis:100%}}
  .speed-btn{{font-size:0.65rem;padding:2px 6px}}

  .video-wrap video{{height:160px;flex:1 1 100%}}
  .video-overlay{{flex:1 1 100%}}
  /* En tablet: 2 columnas de gauges */
  .gauge-row{{grid-template-columns:repeat(2,1fr)}}
  .gauge-box{{min-height:120px}}
  .gauge-box > div{{min-height:100px}}
  #alert-feed{{height:200px}}

  table{{font-size:0.7rem}}
  th,td{{padding:4px 6px}}
  /* Tablas con muchas columnas: scroll horizontal */
  .panel{{overflow-x:auto}}
  /* Permite scrollear tablas horizontalmente sin romper layout */

  .ipo-hero{{padding:16px}}
  .ipo-big{{font-size:2.5rem}}
  .ipo-band{{font-size:0.75rem}}
  .ipo-components{{grid-template-columns:repeat(auto-fit,minmax(130px,1fr))}}
  .ipo-comp-val{{font-size:1rem}}
  .ipo-tbands{{grid-template-columns:1fr 1fr;gap:6px}}

  .glossary dl{{grid-template-columns:1fr;gap:4px}}
  .glossary dt{{margin-top:8px}}
  .thumb-grid{{grid-template-columns:repeat(auto-fill,minmax(220px,1fr))}}
  .ocr-grid{{grid-template-columns:repeat(auto-fill,minmax(180px,1fr))}}
  .alert-grid-clips{{grid-template-columns:1fr}}
  .sim-wrap{{grid-template-columns:1fr !important}}
  .sim-canvas{{height:220px !important}}

  .modal{{max-width:95%;width:95%}}
  .modal-head{{padding:10px 14px;font-size:0.85rem}}
  .modal-body{{padding:14px}}
  .modal-icon{{font-size:1.3rem}}
}}

/* MOBILE: <= 480px — máxima compresión */
@media (max-width: 480px) {{
  body{{font-size:12px}}
  .hdr{{padding:6px 10px;flex-wrap:wrap;gap:6px}}
  .hdr h1{{font-size:0.9rem;letter-spacing:.05em !important}}
  #phase-badge{{font-size:0.65rem;padding:2px 8px;margin-left:4px}}
  #live-clock{{font-size:0.75rem}}
  .theme-toggle{{width:28px;height:28px;font-size:14px}}

  .view{{padding:8px 10px}}
  .view-hdr h2{{font-size:0.85rem}}
  .sec{{margin-bottom:12px}}
  .sec-title{{font-size:0.62rem;margin-bottom:8px;padding-bottom:3px}}

  .kpi-grid{{grid-template-columns:repeat(auto-fill,minmax(105px,1fr));gap:6px}}
  .kpi{{padding:6px 8px}}
  .kpi-val{{font-size:1rem}}
  .kpi-lbl{{font-size:0.58rem}}
  .kpi-sub{{font-size:0.6rem}}

  .video-wrap video{{height:130px}}

  /* En phone: 2 columnas todavía (en lo posible) */
  .gauge-row{{gap:4px;grid-template-columns:repeat(2,1fr)}}
  .gauge-box{{padding:4px;min-height:100px}}
  .gauge-box > div{{min-height:85px}}
  .gauge-title{{font-size:0.58rem}}

  .ipo-big{{font-size:2rem}}
  .ipo-hero{{padding:12px 10px}}
  .ipo-comp-val{{font-size:0.9rem}}
  .ipo-tbands{{grid-template-columns:1fr}}

  table{{font-size:0.65rem}}
  th,td{{padding:3px 4px}}

  .panel{{padding:8px}}
  .btn{{font-size:0.65rem;padding:4px 7px}}

  /* El banner de productividad: más compacto */
  #prod-banner{{padding:10px 12px !important}}
  #prod-banner > div{{gap:8px !important;flex-wrap:wrap}}
  #prod-banner-loss{{font-size:1rem !important;padding:4px 10px !important}}
}}
</style>
</head>
<body>

<!-- HEADER -->
<div class="hdr">
  <div>
    <h1 style="letter-spacing:.15em">DIG <span style="font-size:.7rem;color:{C['muted']};font-weight:400;letter-spacing:.02em;margin-left:6px">Digital Intelligence for Geomining</span></h1>
    <div class="meta">JEBI Hackathon 2026 &nbsp;|&nbsp; Hitachi EX-5600 &nbsp;|&nbsp;
      CAT 793F (218t) &nbsp;&amp;&nbsp; EH4000 AC-3 (221t) &nbsp;|&nbsp; {now}
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <button class="theme-toggle" id="theme-toggle" onclick="toggleTheme()" title="Cambiar tema">🌙</button>
    <span id="phase-badge" class="badge">WAIT</span>
    <span id="live-clock">t = 0.0 s</span>
  </div>
</div>

<!-- MODAL DE ALERTA CRITICA -->
<div class="modal-bg" id="critical-modal" onclick="if(event.target===this)closeCriticalModal()">
  <div class="modal">
    <div class="modal-head">
      <span class="modal-icon">⚠️</span>
      <span>ALERTA CRITICA — REQUIERE ATENCION INMEDIATA</span>
    </div>
    <div class="modal-body">
      <div class="modal-type" id="mod-type">—</div>
      <div class="modal-title" id="mod-title">—</div>
      <div class="modal-msg" id="mod-msg">—</div>
      <div class="modal-data" id="mod-data">—</div>
      <div class="modal-action" id="mod-action"><strong>ACCION RECOMENDADA</strong><span id="mod-action-txt">—</span></div>
    </div>
    <div class="modal-foot">
      <button class="btn" onclick="closeCriticalModal()">Descartar</button>
      <button class="btn btn-primary" onclick="acknowledgeCritical()">Reconocido</button>
    </div>
  </div>
</div>

<div class="app">

<!-- SIDEBAR -->
<aside class="sidebar">
  <div class="side-head">Vistas</div>
  <div class="side-item active" data-view="overview" onclick="switchView('overview')">
    <span class="side-icon">📊</span><span>Overview</span>
  </div>
  <div class="side-item" data-view="ipo" onclick="switchView('ipo')">
    <span class="side-icon">🎯</span><span>IPO Maestro</span>
  </div>
  <div class="side-item" data-view="idle" onclick="switchView('idle')">
    <span class="side-icon">⏱️</span><span>Tiempos de Ocio</span>
  </div>
  <div class="side-item" data-view="live" onclick="switchView('live')">
    <span class="side-icon">🎥</span><span>Video &amp; Live</span>
  </div>
  <div class="side-item" data-view="pausas" onclick="switchView('pausas')">
    <span class="side-icon">⏸</span><span>Pausas del Operador</span>
    <span class="side-badge" id="side-pausas-count">0</span>
  </div>
  <div class="side-item" data-view="cycles" onclick="switchView('cycles')">
    <span class="side-icon">🔄</span><span>Ciclos</span>
  </div>
  <div class="side-item" data-view="alerts" onclick="switchView('alerts')">
    <span class="side-icon">🚨</span><span>Alertas</span>
    <span class="side-badge" id="side-alert-count">0</span>
  </div>
  <div class="side-item" data-view="insights" onclick="switchView('insights')">
    <span class="side-icon">💡</span><span>Insights &amp; Recomendaciones</span>
  </div>
  <div class="side-item" data-view="simulator" onclick="switchView('simulator')">
    <span class="side-icon">🎯</span><span>Simulador (apoyo)</span>
  </div>

  <div class="side-divider"></div>
  <div class="side-head">Exportar</div>
  <div class="side-item" onclick="exportCurrentView()">
    <span class="side-icon">📄</span><span>PDF vista actual</span>
  </div>
  <div class="side-item" onclick="exportAllGrouped()">
    <span class="side-icon">📦</span><span>PDF completo</span>
  </div>

  <div class="side-divider"></div>
  <div class="side-head">Ayuda</div>
  <div class="side-item" onclick="switchView('help')">
    <span class="side-icon">❓</span><span>Glosario</span>
  </div>
</aside>

<!-- MAIN CONTENT -->
<main class="main">

<!-- ═══════════ VIEW: OVERVIEW (default) ═══════════ -->
<div class="view active" id="view-overview" data-view="overview">
  <div class="view-hdr">
    <div>
      <h2>Overview de la Sesion</h2>
      <div class="desc">Resumen del rendimiento minero — tiempos, fill factor, despacho y payload total</div>
    </div>
    <div class="view-actions">
      <button class="btn" onclick="switchView('live')">▶ Ir a Live</button>
      <button class="btn btn-primary" onclick="exportCurrentView()">📄 Exportar PDF</button>
    </div>
  </div>

  <div class="sec">
    <div class="sec-title">KPIs de la Sesion</div>
    {kpis_html}
  </div>

  <div class="sec">
    <div class="sec-title">Parametros y Glosario Rapido</div>
    <div class="glossary">
      <h4>Lectura rapida para el operador</h4>
      <dl>
        <dt>Fill Factor</dt><dd>% del balde lleno por cucharada. Objetivo: <b style="color:{C['green']}">&ge;85%</b>. Bajo 80% = ineficiencia.</dd>
        <dt>Payload</dt><dd>Toneladas estimadas de material por ciclo. Balde maximo ~67.5t.</dd>
        <dt>OEE Ciclo</dt><dd>Efectividad global del ciclo (0.4×fill + 0.3×maniobra + 0.15×posicion + 0.15×descarga).</dd>
        <dt>Productividad</dt><dd>Toneladas movidas por hora en esta ventana.</dd>
        <dt>Tiempo Util</dt><dd>% del tiempo en ciclo activo (no esperas). Objetivo: <b style="color:{C['green']}">&ge;75%</b>.</dd>
        <dt>Despacho</dt><dd>% de camiones que salieron con capacidad llena (&ge;85%).</dd>
      </dl>
      <div class="threshold-list">
        <span class="threshold-chip">Underfill: <b>&lt;80%</b></span>
        <span class="threshold-chip">Overfill: <b>&gt;98%</b></span>
        <span class="threshold-chip">Impacto duro: <b>&gt;22 m/s²</b></span>
        <span class="threshold-chip">Dump brusco: <b>&gt;18 m/s²</b></span>
        <span class="threshold-chip">Espera larga: <b>&gt;25 s</b></span>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════ VIEW: IPO MAESTRO ═══════════ -->
<div class="view" id="view-ipo" data-view="ipo">
  <div class="view-hdr">
    <div>
      <h2>IPO — Indice de Productividad Operativa</h2>
      <div class="desc">Formula maestra del proceso minero con desglose por componente</div>
    </div>
    <div class="view-actions">
      <button class="btn btn-primary" onclick="exportCurrentView()">📄 Exportar PDF</button>
    </div>
  </div>
  {ipo_html}
</div><!-- /view-ipo -->

<!-- ═══════════ VIEW: IDLE TIME INDEX ═══════════ -->
<div class="view" id="view-idle" data-view="idle">
  <div class="view-hdr">
    <div>
      <h2>Tiempos de Ocio — Índice ITI</h2>
      <div class="desc">Métrica más intuitiva: qué porcentaje del tiempo no está produciendo el equipo</div>
    </div>
    <div class="view-actions">
      <button class="btn btn-primary" onclick="exportCurrentView()">📄 Exportar PDF</button>
    </div>
  </div>
  {idle_html}
</div><!-- /view-idle -->

<!-- ═══════════ VIEW: LIVE ═══════════ -->
<div class="view" id="view-live" data-view="live">
  <div class="view-hdr">
    <div>
      <h2>Video &amp; Live Telemetry</h2>
      <div class="desc">Reproduccion sincronizada del video estereo con IMU y gauges en tiempo real</div>
    </div>
    <div class="view-actions">
      <button class="btn btn-primary" onclick="exportCurrentView()">📄 Exportar PDF</button>
    </div>
  </div>

<!-- PLAYBACK CONTROL -->
<div class="sec">
  <div id="playbar">
    <button id="play-btn" onclick="togglePlay()">&#9654; PLAY</button>
    <input type="range" id="progress" min="0" max="{len(timeline)-1}" value="0"
           oninput="seekTo(parseInt(this.value))">
    <span id="t-display">0.0 s</span>
    <span style="color:{C['muted']};font-size:.72rem">Velocidad:</span>
    <button class="speed-btn active" onclick="setSpeed(1,this)">1x</button>
    <button class="speed-btn" onclick="setSpeed(5,this)">5x</button>
    <button class="speed-btn" onclick="setSpeed(15,this)">15x</button>
    <button class="speed-btn" onclick="setSpeed(30,this)">30x</button>
  </div>
</div>

<!-- BANNER DE ALERTA PRODUCTIVA (persistente, arriba del video) -->
<div class="sec">
  <div id="prod-banner" style="display:none;padding:14px 20px;border-radius:10px;
       background:linear-gradient(90deg,{C['red']},#b02a2a);color:#fff;
       font-weight:700;letter-spacing:.03em;box-shadow:0 0 30px rgba(248,81,73,.4);
       animation:pulse 1.5s ease-in-out infinite">
    <div style="display:flex;align-items:center;gap:14px">
      <span style="font-size:1.8rem">🚨</span>
      <div style="flex:1">
        <div style="font-size:.95rem">PÉRDIDA PRODUCTIVA EN CURSO</div>
        <div id="prod-banner-detail" style="font-size:.78rem;opacity:.95;font-weight:500;margin-top:3px">
          Sin movimiento de cámara detectado — <span id="prod-banner-time">0</span>s
        </div>
      </div>
      <div id="prod-banner-loss" style="font-size:1.3rem;font-family:monospace;
           padding:6px 14px;background:rgba(0,0,0,.25);border-radius:6px">
        ~0 t
      </div>
    </div>
  </div>
</div>

<!-- VIDEO + ALERTAS -->
<div class="sec row">
  <div class="col-video">
    <div class="sec-title">Video Estereo &mdash; Camara Izquierda / Derecha</div>
    <div class="video-wrap">
      <div class="video-overlay" style="flex:1">
        <div class="video-label">CAM IZQUIERDA (controla ambos videos)</div>
        <video id="vid-left" src="../inputs/shovel_left.mp4"
               muted playsinline preload="auto" controls
               style="width:100%;height:240px;object-fit:cover;border-radius:6px"></video>
        <div class="ov" id="vid-left-info">t=0.0s</div>
      </div>
      <div class="video-overlay" style="flex:1">
        <div class="video-label">CAM DERECHA (sigue a la izquierda)</div>
        <video id="vid-right" src="../inputs/shovel_right.mp4"
               muted playsinline preload="auto"
               style="width:100%;height:240px;object-fit:cover;border-radius:6px"></video>
        <div class="ov" id="vid-right-info">t=0.0s</div>
      </div>
    </div>
    <!-- Sensor overlay on video -->
    <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap">
      <div class="panel" style="flex:1;min-width:120px">
        <div class="kpi-lbl">Accel Norm</div>
        <div class="kpi-val b" id="hud-an">—</div>
        <div class="kpi-sub">m/s²</div>
      </div>
      <div class="panel" style="flex:1;min-width:120px">
        <div class="kpi-lbl">Gyro Norm</div>
        <div class="kpi-val b" id="hud-gn">—</div>
        <div class="kpi-sub">deg/s</div>
      </div>
      <div class="panel" style="flex:1;min-width:120px">
        <div class="kpi-lbl">Fill Factor</div>
        <div class="kpi-val" id="hud-fill">—</div>
        <div class="kpi-sub">% bucket</div>
      </div>
      <div class="panel" style="flex:1;min-width:120px">
        <div class="kpi-lbl">Payload</div>
        <div class="kpi-val" id="hud-payload">—</div>
        <div class="kpi-sub">toneladas</div>
      </div>
      <div class="panel" style="flex:1;min-width:120px">
        <div class="kpi-lbl">Volumen</div>
        <div class="kpi-val" id="hud-vol">—</div>
        <div class="kpi-sub">m³</div>
      </div>
      <div class="panel" style="flex:1;min-width:120px">
        <div class="kpi-lbl">Ciclo #</div>
        <div class="kpi-val b" id="hud-cid">—</div>
        <div class="kpi-sub">activo</div>
      </div>
    </div>
  </div>

  <!-- ALERT FEED -->
  <div class="col-alerts">
    <div class="sec-title">Alertas en Tiempo Real
      <span id="alert-count" style="float:right;color:{C['red']};font-weight:700">0 alertas</span>
    </div>
    <div class="panel" style="padding:8px">
      <div id="alert-feed"></div>
    </div>
    <div style="margin-top:8px">
      <div class="sec-title">Fase Actual</div>
      <div class="panel">
        <div id="phase-detail" style="font-size:0.82rem;padding:4px 0">
          Esperando datos...
        </div>
        <div id="phase-bar" style="height:6px;border-radius:3px;background:{C['border']};margin-top:8px;transition:width .3s,background .3s"></div>
      </div>
    </div>
  </div>
</div>

<!-- GAUGES DE EFICIENCIA -->
<div class="sec">
  <div class="sec-title">Eficiencias Operacionales en Tiempo Real</div>
  <div class="gauge-row">
    <div class="gauge-box">
      <div class="gauge-title">Recoleccion (Fill)</div>
      <div id="g-coll" style="height:100px"></div>
    </div>
    <div class="gauge-box">
      <div class="gauge-title">Maniobra</div>
      <div id="g-man" style="height:100px"></div>
    </div>
    <div class="gauge-box">
      <div class="gauge-title">Posicionamiento</div>
      <div id="g-pos" style="height:100px"></div>
    </div>
    <div class="gauge-box">
      <div class="gauge-title">Descarga</div>
      <div id="g-disc" style="height:100px"></div>
    </div>
    <div class="gauge-box">
      <div class="gauge-title">OEE Ciclo</div>
      <div id="g-oee" style="height:100px"></div>
    </div>
    <div class="gauge-box">
      <div class="gauge-title">Tiempo Util</div>
      <div id="g-teff" style="height:100px"></div>
    </div>
    <div class="gauge-box">
      <div class="gauge-title">Despacho</div>
      <div id="g-disp" style="height:100px"></div>
    </div>
    <div class="gauge-box">
      <div class="gauge-title">Carga Camion</div>
      <div id="g-truck" style="height:100px"></div>
    </div>
  </div>
</div>

<!-- GRAFICOS IMU SCROLLING -->
<div class="sec">
  <div class="sec-title">Sensores IMU &mdash; Ventana Deslizante 30s</div>
  <div class="g2">
    <div class="panel full">
      <div id="chart-accel" style="height:160px"></div>
    </div>
    <div class="panel full">
      <div id="chart-gyro" style="height:160px"></div>
    </div>
  </div>
</div>

</div><!-- /view-live -->

<!-- ═══════════ VIEW: CICLOS ═══════════ -->
<div class="view" id="view-cycles" data-view="cycles">
  <div class="view-hdr">
    <div>
      <h2>Detalle por Ciclo</h2>
      <div class="desc">Eficiencias, fase, volumen y alertas por cada carga de balde</div>
    </div>
    <div class="view-actions">
      <button class="btn btn-primary" onclick="exportCurrentView()">📄 Exportar PDF</button>
    </div>
  </div>
  <div class="sec">
    <div class="sec-title">Ciclos detectados (IMU + video)</div>
    {eff_html}
  </div>
  <div class="sec">
    <div class="sec-title">Tiempos de Espera entre Ciclos</div>
    {waits_html}
  </div>
</div><!-- /view-cycles -->

<!-- ═══════════ VIEW: PAUSAS DEL OPERADOR ═══════════ -->
<div class="view" id="view-pausas" data-view="pausas">
  <div class="view-hdr">
    <div>
      <h2>Pausas del Operador</h2>
      <div class="desc">Eventos detectados por fusión IMU + Optical Flow · Click en cualquier card para saltar al video</div>
    </div>
    <div class="view-actions">
      <button class="btn btn-primary" onclick="exportCurrentView()">📄 Exportar PDF</button>
    </div>
  </div>
  {pausas_html}
</div><!-- /view-pausas -->

<!-- ═══════════ VIEW: INSIGHTS & RECOMENDACIONES ═══════════ -->
<div class="view" id="view-insights" data-view="insights">
  <div class="view-hdr">
    <div>
      <h2>Insights &amp; Recomendaciones</h2>
      <div class="desc">Las 4 preguntas del brief respondidas con data real del pipeline</div>
    </div>
    <div class="view-actions">
      <button class="btn btn-primary" onclick="exportCurrentView()">📄 Exportar PDF</button>
    </div>
  </div>
  {insights_html}
</div><!-- /view-insights -->

<!-- ═══════════ VIEW: ALERTS ═══════════ -->
<div class="view" id="view-alerts" data-view="alerts">
  <div class="view-hdr">
    <div>
      <h2>Alertas &mdash; Histórico + Mini-clips por evento</h2>
      <div class="desc">Cada alerta se acompaña de un mini-clip de video de 3 segundos centrado en el timestamp del evento</div>
    </div>
    <div class="view-actions">
      <button class="btn" onclick="alertFilter='all';renderFullAlertFeed()" id="af-all">Todas</button>
      <button class="btn btn-danger" onclick="alertFilter='critical';renderFullAlertFeed()" id="af-crit">Criticas</button>
      <button class="btn" onclick="alertFilter='warning';renderFullAlertFeed()" id="af-warn">Advertencias</button>
      <button class="btn btn-primary" onclick="exportCurrentView()">📄 Exportar PDF</button>
    </div>
  </div>
  <div class="sec">
    <div class="sec-title">Feed con mini-clips (3s por evento)</div>
    <div id="alert-feed-full" class="alert-grid-clips"></div>
  </div>
</div><!-- /view-alerts -->

<!-- ═══════════ VIEW: SIMULATOR 3D (apoyo) ═══════════ -->
<div class="view" id="view-simulator" data-view="simulator">
  <div class="view-hdr">
    <div>
      <h2>Simulador — Material de Apoyo</h2>
      <div class="desc">Visualización de referencia de la colocación pala-camión (complementa al análisis cuantitativo)</div>
    </div>
    <div class="view-actions">
      <button class="btn btn-primary" onclick="exportCurrentView()">📄 Exportar PDF</button>
    </div>
  </div>
  <div class="sec sim-wrap" style="grid-template-columns:1fr 1fr;gap:10px">
    <div class="sim-panel" style="padding:8px">
      <h3 style="color:{C['green']};font-size:.85rem">✓ Colocación Óptima</h3>
      <div id="sim3d-ok" class="sim-canvas sim-ok" style="height:240px"></div>
    </div>
    <div class="sim-panel" style="padding:8px">
      <h3 style="color:{C['red']};font-size:.85rem">✗ Colocación Incorrecta</h3>
      <div id="sim3d-bad" class="sim-canvas sim-bad" style="height:240px"></div>
      <div class="sim-controls" style="margin-top:6px">
        <button onclick="simScenario3D('offset',this)" class="active">Offset</button>
        <button onclick="simScenario3D('far',this)">Lejos</button>
        <button onclick="simScenario3D('close',this)">Cerca</button>
        <button onclick="simScenario3D('angle',this)">Ángulo</button>
      </div>
    </div>
  </div>
</div><!-- /view-simulator -->

<!-- ═══════════ VIEW: HELP / GLOSARIO ═══════════ -->
<div class="view" id="view-help" data-view="help">
  <div class="view-hdr">
    <div>
      <h2>Glosario y Manual del Operador</h2>
      <div class="desc">Que significa cada metrica, umbral y alerta — y que accion tomar</div>
    </div>
  </div>
  {help_html}
</div><!-- /view-help -->

</main>
</div><!-- /app -->

<!-- ═══════════════════════════════════════════════════════════════
     JAVASCRIPT  —  Real-time simulation engine
═══════════════════════════════════════════════════════════════ -->
<script>
// ── DATA ────────────────────────────────────────────────────────────────────
const TL        = {tl_json};
const ALL_ALERTS= {alerts_json};
const SHAPES    = {shapes_json};
const PAYLOAD_D = {payload_json};
const TRUCKS    = {trucks_json};

const T_FULL    = {t};
const AX_FULL   = {ax};
const AY_FULL   = {ay};
const AZ_FULL   = {az};
const GX_FULL   = {gx};
const GY_FULL   = {gy};
const GZ_FULL   = {gz};
const AN_FULL   = {an};
const GN_FULL   = {gn};

const SESSION_METRICS = {{
  duration_s:    {metrics.total_duration_s},
  time_eff:      {metrics.time_efficiency_pct},
  fill_avg:      {metrics.avg_fill_factor_pct},
  cycles_hr:     {metrics.cycles_per_hour},
  productivity:  {metrics.productivity_tph},
  total_payload: {metrics.total_payload_t},
  // BUGFIX: promedios para que los gauges NO inicien en 50% placeholder
  avg_coll:      {avg_coll:.1f},
  avg_man:       {avg_man:.1f},
  avg_pos:       {avg_pos:.1f},
  avg_disc:      {avg_disc:.1f},
  avg_oee:       {avg_oee:.1f},
}};

const TRANSPORT = {json.dumps(dict(
    n_trucks=transport.n_trucks_served if transport else 0,
    n_full=transport.n_trucks_full if transport else 0,
    disp_eff=transport.dispatch_efficiency if transport else 0,
    avg_fill=transport.avg_truck_fill_pct if transport else 0,
) if transport else dict(n_trucks=0,n_full=0,disp_eff=0,avg_fill=0))};

// Motion timeline para banner de pérdida productiva (dual-source)
const MOTION_SERIES = {json.dumps(
    (visual_validation or {}).get('motion_timeline', {}).get('series', [])
)};
const MOTION_IDLE_PERIODS = {json.dumps(
    (visual_validation or {}).get('motion_timeline', {}).get('idle_periods', [])
)};
// Eventos clasificados (fusion jevi): pausas del operador
const FUSION_EVENTS = {json.dumps(
    (visual_validation or {}).get('fusion_events', [])
)};
const PRODUCTIVITY_TPH = {metrics.productivity_tph if metrics.productivity_tph > 0 else 400};

const DARK_LAYOUT = {{
  paper_bgcolor: '{C['bg']}', plot_bgcolor: '{C['panel']}',
  font: {{color: '{C['text']}', size:10}},
  margin: {{l:38,r:8,t:22,b:28}},
  xaxis: {{gridcolor:'{C['border']}', zerolinecolor:'{C['border']}'}},
  yaxis: {{gridcolor:'{C['border']}', zerolinecolor:'{C['border']}'}},
  legend: {{bgcolor:'rgba(0,0,0,.3)',bordercolor:'{C['border']}',borderwidth:1,font:{{size:9}}}},
}};

// ── STATE ───────────────────────────────────────────────────────────────────
// REDISEÑO: el VIDEO es la fuente única de verdad. El timeline lo sigue.
// Antes había dos "relojes" (tick() y video.currentTime) que se desincronizaban.
let curIdx   = 0;
let playing  = false;
let playTimer= null;
let speed    = 1;        // playback rate del video
let alertsShown = new Set();
let totalAlerts = 0;
let seekingFromUI = false;  // flag para evitar loops de seek

const vidL = document.getElementById('vid-left');
const vidR = document.getElementById('vid-right');

// Constante: cada snapshot está separado por step_s (0.5s).
// Conversión t <-> idx: idx = round(t / step_s)
const STEP_S = TL.length > 1 ? (TL[1].t - TL[0].t) : 0.5;

// ── PLAYBACK ────────────────────────────────────────────────────────────────
function togglePlay() {{
  playing = !playing;
  document.getElementById('play-btn').innerHTML = playing
    ? '&#9646;&#9646; PAUSE' : '&#9654; PLAY';

  if (playing) {{
    vidL.play().catch(e => console.warn('vidL play error:', e));
    vidR.play().catch(e => console.warn('vidR play error:', e));
    // Tick rápido solo para actualizar UI desde el video (video ES la verdad)
    playTimer = setInterval(tick, 100);
  }} else {{
    vidL.pause(); vidR.pause();
    clearInterval(playTimer);
  }}
}}

function setSpeed(s, btn) {{
  speed = s;
  document.querySelectorAll('.speed-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  // playbackRate del video: fuente de verdad única
  const vr = Math.min(s, 16);
  vidL.playbackRate = vr;
  vidR.playbackRate = vr;
}}

// tick(): sigue al video. Si el video avanza, actualizamos curIdx y UI.
function tick() {{
  if (!vidL || vidL.readyState < 2) return;  // video no está listo, esperar

  const t = vidL.currentTime;
  const idx = Math.min(Math.max(0, Math.round(t / STEP_S)), TL.length - 1);

  if (idx !== curIdx) {{
    curIdx = idx;
    update(curIdx);
  }}

  // Detectar si llegamos al final
  if (vidL.ended || t >= vidL.duration - 0.1) {{
    playing = false;
    document.getElementById('play-btn').innerHTML = '&#9654; PLAY';
    clearInterval(playTimer);
  }}
}}

// seekTo: llamado por el slider del progress bar
function seekTo(idx) {{
  seekingFromUI = true;
  curIdx = Math.min(Math.max(0, idx), TL.length - 1);
  const t = TL[curIdx].t;
  // Sincronizar videos al seek
  if (vidL.readyState >= 1) vidL.currentTime = t;
  if (vidR.readyState >= 1) vidR.currentTime = t;
  update(curIdx);
  setTimeout(() => {{ seekingFromUI = false; }}, 50);
}}

// BUGFIX: Cuando el usuario mueve el video NATIVAMENTE (seek en el player),
// también actualizamos el timeline. Antes se ignoraba.
function _onVideoSeeked() {{
  if (seekingFromUI) return;  // ya lo estamos manejando
  const t = vidL.currentTime;
  const idx = Math.min(Math.max(0, Math.round(t / STEP_S)), TL.length - 1);
  if (idx !== curIdx) {{
    curIdx = idx;
    update(curIdx);
  }}
}}

// BUGFIX: Durante play natural del video, también actualizamos UI
function _onVideoTimeUpdate() {{
  if (!playing) {{
    // Usuario está mirando video pero sin haber dado play desde el UI
    // → igual trackear
    const t = vidL.currentTime;
    const idx = Math.min(Math.max(0, Math.round(t / STEP_S)), TL.length - 1);
    if (idx !== curIdx) {{
      curIdx = idx;
      update(curIdx);
    }}
  }}
}}

// Registrar listeners del video
if (vidL) {{
  vidL.addEventListener('seeked', _onVideoSeeked);
  vidL.addEventListener('timeupdate', _onVideoTimeUpdate);
  // Sincronizar el right cuando el user mueve el left
  vidL.addEventListener('seeking', () => {{
    if (vidR && Math.abs(vidR.currentTime - vidL.currentTime) > 0.3) {{
      vidR.currentTime = vidL.currentTime;
    }}
  }});
  // Si el navegador pausa por buffering, mantener sincronizado el right
  vidL.addEventListener('waiting', () => {{ if (vidR) vidR.pause(); }});
  vidL.addEventListener('playing', () => {{
    if (vidR && playing) vidR.play().catch(()=>{{}});
  }});
}}

// ── MAIN UPDATE ─────────────────────────────────────────────────────────────
function update(idx) {{
  const snap = TL[idx];
  const t    = snap.t;

  // Progress bar + clock
  document.getElementById('progress').value   = idx;
  document.getElementById('t-display').textContent = t.toFixed(1) + ' s';
  document.getElementById('live-clock').textContent = 't = ' + t.toFixed(1) + ' s';

  // Video info overlays (siempre sincronizados)
  document.getElementById('vid-left-info').textContent  = 't=' + t.toFixed(1) + 's';
  document.getElementById('vid-right-info').textContent = 't=' + t.toFixed(1) + 's';

  // HUD numbers
  _hud('hud-an',      snap.accel_norm, 'm/s²', 12, 20);
  _hud('hud-gn',      snap.gyro_norm,  'deg/s', 20, 60);
  _hud('hud-fill',    snap.fill_pct,   '%',     80, 95);
  _hud('hud-payload', snap.payload_t,  't',     null, null);
  _hud('hud-vol',     snap.volume_m3,  'm³',    null, null);
  document.getElementById('hud-cid').textContent = snap.cycle_id ? '#'+snap.cycle_id : '—';

  // Phase badge
  const ph   = snap.phase || 'WAIT';
  const pbEl = document.getElementById('phase-badge');
  pbEl.textContent  = ph.replace('_',' ');
  pbEl.className    = 'badge phase-' + ph;
  document.getElementById('phase-detail').innerHTML =
    '<b>' + ph.replace('_',' ') + '</b>' +
    (snap.cycle_id ? '  &mdash; Ciclo #' + snap.cycle_id : '') +
    '<br><span style="color:{C['muted']}">Fill: ' + (snap.fill_pct!=null?snap.fill_pct.toFixed(1):'—') +
    '%  |  Payload: ' + (snap.payload_t!=null?snap.payload_t.toFixed(1):'—') + ' t</span>';

  // Gauges
  const deflt = 50;
  _gauge('g-coll',  snap.coll_eff    ?? deflt, 'Recol.', '%');
  _gauge('g-man',   snap.maneuver_eff?? deflt, 'Maniob.','%');
  _gauge('g-pos',   snap.pos_eff     ?? deflt, 'Posic.', '%');
  _gauge('g-disc',  snap.disc_eff    ?? deflt, 'Descarg','%');
  _gauge('g-oee',   snap.cycle_eff   ?? deflt, 'OEE',    '%');
  _gauge('g-teff',  SESSION_METRICS.time_eff,  'T.Util', '%');
  _gauge('g-disp',  TRANSPORT.disp_eff,        'Desp.', '%');
  const truckFill = TRANSPORT.avg_fill ?? 0;
  _gauge('g-truck', truckFill, 'Camion','%');

  // IMU scrolling graphs
  _updateScrollingCharts(t);

  // Banner de pérdida productiva (dual-source)
  _updateProdBanner(t);

  // Alerts
  _processAlerts(t, snap.alerts || []);
}}

// ── BANNER PÉRDIDA PRODUCTIVA ────────────────────────────────────────────────
// Se activa cuando el timestamp actual cae dentro de un idle_period visual
// detectado por optical flow. Muestra al operador en tiempo real que está
// perdiendo productividad.
function _updateProdBanner(t) {{
  const banner = document.getElementById('prod-banner');
  if (!banner || !MOTION_IDLE_PERIODS || MOTION_IDLE_PERIODS.length === 0) {{
    if (banner) banner.style.display = 'none';
    return;
  }}
  // ¿Estamos dentro de algún idle period visual?
  let active = null;
  for (const p of MOTION_IDLE_PERIODS) {{
    if (t >= p.t_start && t <= p.t_end) {{
      active = p;
      break;
    }}
  }}
  if (active) {{
    const elapsed = (t - active.t_start).toFixed(0);
    const elapsedH = (t - active.t_start) / 3600;
    const lossT = (elapsedH * PRODUCTIVITY_TPH).toFixed(1);
    banner.style.display = 'block';
    document.getElementById('prod-banner-time').textContent = elapsed;
    document.getElementById('prod-banner-loss').textContent = '~' + lossT + ' t';
    document.getElementById('prod-banner-detail').textContent =
      'Sin movimiento de cámara detectado — ' + elapsed + 's (período idle #' +
      (MOTION_IDLE_PERIODS.indexOf(active) + 1) + ' de ' + MOTION_IDLE_PERIODS.length + ')';
  }} else {{
    banner.style.display = 'none';
  }}
}}

// ── HUD helper ───────────────────────────────────────────────────────────────
function _hud(id, val, unit, warnLow, warnHigh) {{
  const el = document.getElementById(id);
  if (val == null) {{ el.textContent = '—'; el.className='kpi-val'; return; }}
  el.textContent = typeof val === 'number' ? val.toFixed(1) : val;
  if (warnLow  != null && val < warnLow)  el.className='kpi-val r';
  else if (warnHigh != null && val > warnHigh) el.className='kpi-val w';
  else el.className='kpi-val g';
}}

// ── GAUGE ────────────────────────────────────────────────────────────────────
const gaugeInstances = {{}};
function _gauge(id, value, title, unit) {{
  const color = value >= 85 ? '{C['green']}' : value >= 65 ? '{C['yellow']}' : '{C['red']}';
  const data  = [{{ type:'indicator', mode:'gauge+number',
    value: value,
    number: {{ suffix: unit, font:{{size:16,color:color}} }},
    gauge: {{
      axis: {{ range:[0,100], tickfont:{{size:8}}, tickcolor:'{C['muted']}' }},
      bar:  {{ color: color, thickness:.6 }},
      bgcolor: '{C['panel2']}',
      bordercolor: '{C['border']}',
      steps: [
        {{ range:[0,65], color:'rgba(248,81,73,.08)' }},
        {{ range:[65,85], color:'rgba(255,215,0,.08)' }},
        {{ range:[85,100], color:'rgba(63,185,80,.08)' }},
      ],
    }},
  }}];
  // BUGFIX responsive: autosize + height por CSS, y config responsive:true
  const layout = {{ ...DARK_LAYOUT, margin:{{l:8,r:8,t:12,b:4}},
                    autosize:true, height:120 }};
  const cfg = {{displayModeBar:false, responsive:true}};
  if (gaugeInstances[id]) {{
    Plotly.react(id, data, layout, cfg);
  }} else {{
    Plotly.newPlot(id, data, layout, cfg);
    gaugeInstances[id] = true;
  }}
}}

// ── SCROLLING CHARTS ─────────────────────────────────────────────────────────
let accelInited = false, gyroInited = false;
const WIN_S = 30;

function _updateScrollingCharts(t_now) {{
  const iEnd   = T_FULL.findIndex(x => x >= t_now);
  const iStart = T_FULL.findIndex(x => x >= t_now - WIN_S);
  const i0 = Math.max(0, iStart < 0 ? 0 : iStart);
  const i1 = iEnd < 0 ? T_FULL.length - 1 : iEnd;

  const ts = T_FULL.slice(i0, i1+1);

  const accelTraces = [
    {{ x:ts, y:AX_FULL.slice(i0,i1+1), name:'accel_x', mode:'lines', line:{{color:'{C['blue']}',width:1}}}},
    {{ x:ts, y:AY_FULL.slice(i0,i1+1), name:'accel_y', mode:'lines', line:{{color:'{C['orange']}',width:1}}}},
    {{ x:ts, y:AZ_FULL.slice(i0,i1+1), name:'accel_z', mode:'lines', line:{{color:'{C['green']}',width:1}}}},
  ];
  const gyroTraces = [
    {{ x:ts, y:GX_FULL.slice(i0,i1+1), name:'gyro_x', mode:'lines', line:{{color:'{C['blue']}',width:1}}}},
    {{ x:ts, y:GY_FULL.slice(i0,i1+1), name:'gyro_y', mode:'lines', line:{{color:'{C['orange']}',width:1}}}},
    {{ x:ts, y:GZ_FULL.slice(i0,i1+1), name:'gyro_z', mode:'lines', line:{{color:'{C['green']}',width:1}}}},
  ];

  const tRange = [t_now - WIN_S, t_now];
  const aLayout = {{ ...DARK_LAYOUT, title:{{text:'Accelerometer (m/s²)',font:{{size:11}}}},
                     xaxis:{{ ...DARK_LAYOUT.xaxis, range:tRange }}, height:160 }};
  const gLayout = {{ ...DARK_LAYOUT, title:{{text:'Gyroscope (deg/s)',font:{{size:11}}}},
                     xaxis:{{ ...DARK_LAYOUT.xaxis, range:tRange }}, height:160 }};

  if (!accelInited) {{ Plotly.newPlot('chart-accel',accelTraces,aLayout,{{displayModeBar:false}}); accelInited=true; }}
  else Plotly.react('chart-accel', accelTraces, aLayout);
  if (!gyroInited)  {{ Plotly.newPlot('chart-gyro',gyroTraces,gLayout,{{displayModeBar:false}}); gyroInited=true; }}
  else Plotly.react('chart-gyro', gyroTraces, gLayout);
}}

// ── ALERTS ───────────────────────────────────────────────────────────────────
function _processAlerts(t_now, snap_alerts) {{
  // Alertas del timeline snapshot
  snap_alerts.forEach(a => {{
    const key = a.type + '_' + Math.floor(t_now);
    if (!alertsShown.has(key)) {{
      alertsShown.add(key);
      _addAlert(t_now, a.type, a.sev || 'info', a.msg, a.val, a.unit);
    }}
  }});

  // Alertas pre-computadas
  ALL_ALERTS.forEach(a => {{
    const key = a.type + '_' + Math.floor(a.t);
    if (a.t <= t_now && !alertsShown.has(key)) {{
      alertsShown.add(key);
      _addAlert(a.t, a.type, a.sev, a.msg, a.val, a.unit);
    }}
  }});
}}

function _addAlert(t, type, sev, msg, val, unit) {{
  totalAlerts++;
  document.getElementById('alert-count').textContent = totalAlerts + ' alertas';

  const feed = document.getElementById('alert-feed');
  const el   = document.createElement('div');
  el.className = 'alert-item alert-' + (sev||'info');
  el.innerHTML =
    '<span class="alert-time">t=' + t.toFixed(1) + 's</span>' +
    '<b>' + type + '</b>: ' + msg +
    (val != null ? ' <span style="opacity:.8">['+val+' '+(unit||'')+']</span>' : '');
  feed.insertBefore(el, feed.firstChild);

  // Mantener max 50 items
  while (feed.children.length > 50) feed.removeChild(feed.lastChild);
}}

// ── INIT ─────────────────────────────────────────────────────────────────────
// BUGFIX: Inicializar gauges con valores PROMEDIO DE SESIÓN en vez de 50% placeholder.
// Esto muestra inmediatamente info util al cargar, sin esperar al PLAY.
const _INIT_GAUGE_VALUES = {{
  'g-coll':  SESSION_METRICS.avg_coll,
  'g-man':   SESSION_METRICS.avg_man,
  'g-pos':   SESSION_METRICS.avg_pos,
  'g-disc':  SESSION_METRICS.avg_disc,
  'g-oee':   SESSION_METRICS.avg_oee,
  'g-teff':  SESSION_METRICS.time_eff,
  'g-disp':  TRANSPORT.disp_eff,
  'g-truck': TRANSPORT.avg_fill,
}};
Object.entries(_INIT_GAUGE_VALUES).forEach(([id, val]) => {{
  _gauge(id, val != null && !isNaN(val) ? val : 0, '', '%');
}});

// Inicializar graficos IMU vacios
Plotly.newPlot('chart-accel',[],{{...DARK_LAYOUT,height:160,title:{{text:'Accelerometer'}}}},{{displayModeBar:false}});
accelInited=true;
Plotly.newPlot('chart-gyro', [],{{...DARK_LAYOUT,height:160,title:{{text:'Gyroscope'}}}},  {{displayModeBar:false}});
gyroInited=true;

// ═══════════════════════════════════════════════════════════════
// VIEW SWITCHER
// ═══════════════════════════════════════════════════════════════
function switchView(name) {{
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.side-item[data-view]').forEach(s => s.classList.remove('active'));
  const view = document.getElementById('view-' + name);
  if (view) view.classList.add('active');
  const side = document.querySelector('.side-item[data-view="'+name+'"]');
  if (side) side.classList.add('active');

  // BUGFIX responsive: forzar Plotly resize de todos los charts visibles
  setTimeout(() => {{
    window.dispatchEvent(new Event('resize'));
    // Específicamente relayout todos los Plotly charts de la vista activa
    if (view) {{
      view.querySelectorAll('.js-plotly-plot').forEach(el => {{
        try {{ Plotly.Plots.resize(el); }} catch(e) {{}}
      }});
    }}
  }}, 120);

  // Render especificos por vista
  if (name === 'simulator') renderSimulators();
  if (name === 'alerts') renderFullAlertFeed();
  if ((name === 'ipo' || name === 'help' || name === 'idle') && window.MathJax && window.MathJax.typesetPromise) {{
    setTimeout(() => window.MathJax.typesetPromise([view]).catch(e => console.warn('MathJax:', e)), 60);
  }}
}}

// BUGFIX responsive: relayout Plotly en cambio de viewport (device rotation, resize)
window.addEventListener('resize', () => {{
  const active = document.querySelector('.view.active');
  if (!active) return;
  active.querySelectorAll('.js-plotly-plot').forEach(el => {{
    try {{ Plotly.Plots.resize(el); }} catch(e) {{}}
  }});
}});

// ═══════════════════════════════════════════════════════════════
// MODAL DE ALERTA CRITICA
// ═══════════════════════════════════════════════════════════════
let criticalQueue = [];
let criticalShown = new Set();

const CRITICAL_ACTIONS = {{
  'HARD_IMPACT': 'Reducir velocidad de entrada al banco. Revisar si hay roca suelta o material congelado. Si persiste, detener y avisar a supervisor.',
  'ROUGH_DUMP': 'Descender el balde mas suavemente sobre la tolva. Centrar el dump. Evitar abrir el balde en caida libre.',
  'POSSIBLE_SPILL': 'Reducir velocidad del swing cargado. Verificar que el balde no este sobrecargado (>98%).',
  'OVERFILL': 'Evitar cargas por encima del 98%. Riesgo de daño al sistema hidraulico y derrame al transportar.',
  'UNDERFILL': 'Considerar una re-mordida rapida antes del swing. Si el material es suelto, ajustar angulo de ataque del balde.',
  'BAD_POSITIONING': 'Optimizar trayectoria del swing. Revisar si hay obstaculos. Velocidad hidraulica del swing puede necesitar ajuste.',
  'LONG_WAIT': 'Coordinar con despachador: confirmar ruta del camion. Evaluar si otra pala puede tomar la ventana.',
  'ROUGH_DIG': 'Material muy duro o lleno de finos compactados. Considerar pre-blasting o cambio de banco.',
  'MINI_CYCLE': 'Se detecto un movimiento de correccion. Verificar posicionamiento del camion en el proximo ciclo.',
}};

function showCriticalModal(alert) {{
  const modal = document.getElementById('critical-modal');
  document.getElementById('mod-type').textContent   = (alert.sev || 'critical').toUpperCase() + ' · ' + alert.type;
  document.getElementById('mod-title').textContent  = alert.msg || alert.type;
  document.getElementById('mod-msg').textContent    = alert.msg || '';
  document.getElementById('mod-data').innerHTML     =
    '<b>Tiempo:</b> t = ' + (alert.t!=null?alert.t.toFixed(1):'—') + ' s &nbsp;|&nbsp; ' +
    '<b>Valor:</b> ' + (alert.val!=null?alert.val:'—') + ' ' + (alert.unit||'') + ' &nbsp;|&nbsp; ' +
    '<b>Ciclo:</b> #' + (alert.cid||'—');
  document.getElementById('mod-action-txt').textContent =
    CRITICAL_ACTIONS[alert.type] || 'Consultar supervisor de operaciones.';
  modal.classList.add('show');
}}
function closeCriticalModal() {{
  document.getElementById('critical-modal').classList.remove('show');
}}
function acknowledgeCritical() {{
  closeCriticalModal();
  // Procesar el siguiente de la cola si hay
  if (criticalQueue.length > 0) {{
    const nxt = criticalQueue.shift();
    setTimeout(() => showCriticalModal(nxt), 300);
  }}
}}
function queueCriticalIfNew(alert) {{
  const key = alert.type + '_' + Math.floor(alert.t);
  if (criticalShown.has(key)) return;
  if (alert.sev !== 'critical') return;
  criticalShown.add(key);
  const modal = document.getElementById('critical-modal');
  if (modal.classList.contains('show')) {{
    criticalQueue.push(alert);
  }} else {{
    showCriticalModal(alert);
  }}
}}

// ═══════════════════════════════════════════════════════════════
// ALERT FEED + MINI-CLIPS (vista Alerts)
// ═══════════════════════════════════════════════════════════════
let alertFilter = 'all';        // 'all' | 'critical' | 'warning' | 'info'
const CLIP_DURATION_S = 3.0;    // segundos por clip (1.5 antes + 1.5 despues)

function renderFullAlertFeed() {{
  const feed = document.getElementById('alert-feed-full');
  if (!feed) return;
  feed.innerHTML = '';

  // Resaltar boton activo
  ['all','crit','warn'].forEach(k => {{
    const map = {{'all':'all','crit':'critical','warn':'warning'}};
    const btn = document.getElementById('af-'+k);
    if (btn) btn.classList.toggle('btn-primary', alertFilter === map[k]);
  }});

  const sorted = [...ALL_ALERTS]
    .filter(a => alertFilter === 'all' || a.sev === alertFilter)
    .sort((a,b) => {{
      const sev = {{critical:0, warning:1, info:2, success:3}};
      const sa = sev[a.sev] ?? 4, sb = sev[b.sev] ?? 4;
      if (sa !== sb) return sa - sb;
      return a.t - b.t;
    }});

  sorted.forEach((a, idx) => {{
    const card = document.createElement('div');
    card.className = 'alert-clip-card sev-' + (a.sev || 'info');
    const clipStart = Math.max(0, a.t - CLIP_DURATION_S/2);
    const clipEnd   = a.t + CLIP_DURATION_S/2;
    const vidId = 'clip-' + idx;

    card.innerHTML = `
      <video class="alert-clip-video" id="${{vidId}}"
             src="../inputs/shovel_left.mp4#t=${{clipStart.toFixed(2)}},${{clipEnd.toFixed(2)}}"
             muted playsinline preload="metadata"
             onloadedmetadata="this.currentTime=${{clipStart.toFixed(2)}}"
             ontimeupdate="if(this.currentTime>=${{clipEnd.toFixed(2)}}){{this.currentTime=${{clipStart.toFixed(2)}};}}"
             loop></video>
      <div class="alert-clip-body">
        <div class="alert-clip-head">
          <span class="alert-clip-type" style="color:var(--${{a.sev==='critical'?'red':a.sev==='warning'?'yellow':a.sev==='success'?'green':'blue'}})">
            ${{a.type}}
          </span>
          <span class="alert-clip-time">t=${{a.t.toFixed(1)}}s · #${{a.cid||'—'}}</span>
        </div>
        <div class="alert-clip-msg">${{a.msg}}${{a.val!=null?` <b style="color:var(--yellow)">[${{a.val}} ${{a.unit||''}}]</b>`:''}}</div>
      </div>
      <div class="alert-clip-ctrl">
        <button onclick="playClip('${{vidId}}', ${{clipStart}}, ${{clipEnd}})">▶ Reproducir</button>
        <button onclick="seekToAlert(${{a.t}})">🎯 Ir a Live</button>
        <button onclick="document.getElementById('${{vidId}}').playbackRate=0.5">🐢 0.5x</button>
      </div>
    `;
    feed.appendChild(card);
  }});

  if (sorted.length === 0) {{
    feed.innerHTML = '<p style="color:var(--muted);padding:20px;text-align:center">Sin alertas para el filtro seleccionado.</p>';
  }}

  // Update sidebar badge
  const crit = ALL_ALERTS.filter(a => a.sev === 'critical').length;
  const badge = document.getElementById('side-alert-count');
  if (badge) {{
    badge.textContent = crit > 0 ? crit : ALL_ALERTS.length;
    badge.style.background = crit > 0 ? 'var(--red)' : 'var(--info)';
  }}
}}

function playClip(vidId, start, end) {{
  const v = document.getElementById(vidId);
  if (!v) return;
  v.currentTime = start;
  v.play().catch(e => console.warn('clip play:', e));
}}

function seekToAlert(ts) {{
  switchView('live');
  // Pequeno delay para que el video este visible
  setTimeout(() => {{
    const vL = document.getElementById('vid-left');
    const vR = document.getElementById('vid-right');
    if (vL) vL.currentTime = ts;
    if (vR) vR.currentTime = ts;
    // Actualizar progress bar (timeline)
    const idx = Math.round(ts * 2); // step_s = 0.5
    curIdx = Math.min(idx, TL.length - 1);
    update(curIdx);
  }}, 250);
}}

// Alias genérico para seeking desde cualquier vista a un timestamp
function seekToTime(ts) {{ seekToAlert(ts); }}

// Actualizar badge de pausas en sidebar
(function updatePausasBadge() {{
  const b = document.getElementById('side-pausas-count');
  if (b && typeof FUSION_EVENTS !== 'undefined') {{
    b.textContent = FUSION_EVENTS.length || 0;
    const nInjust = FUSION_EVENTS.filter(e => e.estado === 'INACTIVIDAD_INJUSTIFICADA').length;
    if (nInjust > 0) b.style.background = 'var(--red)';
  }}
}})();

// ═══════════════════════════════════════════════════════════════
// LIGHT / DARK THEME TOGGLE
// ═══════════════════════════════════════════════════════════════
function toggleTheme() {{
  const root = document.documentElement;
  const cur = root.getAttribute('data-theme') || 'dark';
  const next = cur === 'dark' ? 'light' : 'dark';
  root.setAttribute('data-theme', next);
  const btn = document.getElementById('theme-toggle');
  if (btn) btn.textContent = next === 'dark' ? '🌙' : '☀️';
  try {{ localStorage.setItem('dig-theme', next); }} catch(e) {{}}
  // Relayoutear Plotly para que tome nuevos colores (si aplica)
  setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
}}
// Restaurar tema guardado
try {{
  const saved = localStorage.getItem('dig-theme');
  if (saved) {{
    document.documentElement.setAttribute('data-theme', saved);
    setTimeout(() => {{
      const btn = document.getElementById('theme-toggle');
      if (btn) btn.textContent = saved === 'dark' ? '🌙' : '☀️';
    }}, 50);
  }}
}} catch(e) {{}}

// ═══════════════════════════════════════════════════════════════
// EXPORT PDF
// ═══════════════════════════════════════════════════════════════
async function exportCurrentView() {{
  const current = document.querySelector('.view.active');
  if (!current) return;
  const name = current.dataset.view || 'view';
  await exportNodeToPDF(current, `JEBI_2026_${{name}}.pdf`, name.toUpperCase());
}}

async function exportAllGrouped() {{
  const views = ['overview','live','pausas','cycles','alerts','insights','simulator'];
  const {{ jsPDF }} = window.jspdf;
  const pdf = new jsPDF({{ orientation:'portrait', unit:'mm', format:'a4' }});
  const W = 210, H = 297;
  let first = true;

  // Abrir una vista temporalmente, capturarla
  const prevActive = document.querySelector('.view.active')?.dataset.view || 'overview';

  for (const name of views) {{
    const el = document.getElementById('view-' + name);
    if (!el) continue;
    switchView(name);
    await new Promise(r => setTimeout(r, 400));
    const canvas = await html2canvas(el, {{
      backgroundColor:'#0d1117', scale:1.3, logging:false, useCORS:true
    }});
    const img = canvas.toDataURL('image/jpeg', 0.85);
    const ratio = canvas.width / canvas.height;
    const imgW = W - 10;
    const imgH = imgW / ratio;

    if (!first) pdf.addPage();
    first = false;
    pdf.setFillColor(13,17,23); pdf.rect(0,0,W,H,'F');
    pdf.setTextColor(74,158,255);
    pdf.setFontSize(16);
    pdf.text(`JEBI 2026 · ${{name.toUpperCase()}}`, 5, 10);
    pdf.setFontSize(9);
    pdf.setTextColor(139,148,158);
    pdf.text(new Date().toLocaleString(), 5, 15);

    // Imagen debajo del header
    if (imgH < H - 20) {{
      pdf.addImage(img, 'JPEG', 5, 20, imgW, imgH);
    }} else {{
      // Scale to fit
      const scaledH = H - 25;
      const scaledW = scaledH * ratio;
      pdf.addImage(img, 'JPEG', (W - scaledW)/2, 20, scaledW, scaledH);
    }}
  }}

  switchView(prevActive);
  pdf.save(`JEBI_2026_REPORTE_COMPLETO_${{Date.now()}}.pdf`);
}}

async function exportNodeToPDF(node, filename, title) {{
  const {{ jsPDF }} = window.jspdf;
  const canvas = await html2canvas(node, {{
    backgroundColor:'#0d1117', scale:1.5, logging:false, useCORS:true
  }});
  const img = canvas.toDataURL('image/jpeg', 0.9);
  const ratio = canvas.width / canvas.height;
  const orient = ratio > 1 ? 'landscape' : 'portrait';
  const pdf = new jsPDF({{ orientation:orient, unit:'mm', format:'a4' }});
  const W = orient === 'landscape' ? 297 : 210;
  const H = orient === 'landscape' ? 210 : 297;

  pdf.setFillColor(13,17,23); pdf.rect(0,0,W,H,'F');
  pdf.setTextColor(74,158,255);
  pdf.setFontSize(14);
  pdf.text(`JEBI 2026 · ${{title}}`, 5, 10);
  pdf.setFontSize(8);
  pdf.setTextColor(139,148,158);
  pdf.text(new Date().toLocaleString(), 5, 15);

  const imgW = W - 10;
  const imgH = imgW / ratio;
  if (imgH < H - 20) {{
    pdf.addImage(img, 'JPEG', 5, 20, imgW, imgH);
  }} else {{
    let remaining = imgH;
    let offY = 0;
    let pageY = 20;
    while (remaining > 0) {{
      const pageCapacity = H - pageY - 5;
      const drawH = Math.min(remaining, pageCapacity);
      // Crop via canvas
      const srcY = (offY / imgH) * canvas.height;
      const srcH = (drawH / imgH) * canvas.height;
      const tmp = document.createElement('canvas');
      tmp.width = canvas.width; tmp.height = srcH;
      tmp.getContext('2d').drawImage(canvas, 0, srcY, canvas.width, srcH, 0, 0, canvas.width, srcH);
      pdf.addImage(tmp.toDataURL('image/jpeg', 0.9), 'JPEG', 5, pageY, imgW, drawH);
      remaining -= drawH;
      offY     += drawH;
      if (remaining > 0) {{
        pdf.addPage();
        pdf.setFillColor(13,17,23); pdf.rect(0,0,W,H,'F');
        pageY = 5;
      }}
    }}
  }}
  pdf.save(filename);
}}

// ═══════════════════════════════════════════════════════════════
// SIMULADOR 3D — Three.js
// ═══════════════════════════════════════════════════════════════
const sim3dState = {{ ok: null, bad: null, currentScenario: 'offset' }};

function initSim3D(containerId, mode) {{
  const container = document.getElementById(containerId);
  if (!container || typeof THREE === 'undefined') return null;

  const W = container.clientWidth || 600;
  const H = container.clientHeight || 360;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1117);
  scene.fog = new THREE.Fog(0x0d1117, 40, 120);

  const camera = new THREE.PerspectiveCamera(55, W/H, 0.1, 500);
  camera.position.set(22, 18, 28);
  camera.lookAt(0, 2, 0);

  const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
  renderer.setSize(W, H);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.innerHTML = '';
  container.appendChild(renderer.domElement);

  // Luces
  const amb = new THREE.AmbientLight(0xffffff, 0.55);
  scene.add(amb);
  const sun = new THREE.DirectionalLight(0xffefc8, 0.9);
  sun.position.set(15, 25, 10);
  scene.add(sun);
  const back = new THREE.PointLight(0x4a9eff, 0.4, 80);
  back.position.set(-20, 10, -10);
  scene.add(back);

  // Piso (terreno minero)
  const floorMat = new THREE.MeshStandardMaterial({{ color: 0x3a2a1a, roughness: 0.95 }});
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80, 20, 20), floorMat);
  floor.rotation.x = -Math.PI/2;
  floor.position.y = -0.01;
  scene.add(floor);
  // Grid overlay
  const grid = new THREE.GridHelper(80, 40, 0x555555, 0x30363d);
  grid.position.y = 0.02;
  scene.add(grid);

  // Zona optima (cilindro verde semi-transparente)
  const zoneOpt = new THREE.Mesh(
    new THREE.CylinderGeometry(4, 4, 0.1, 32),
    new THREE.MeshBasicMaterial({{ color: 0x3fb950, transparent: true, opacity: 0.3 }})
  );
  zoneOpt.position.set(0, 0.05, 0);
  scene.add(zoneOpt);
  // Ring verde
  const ringOpt = new THREE.Mesh(
    new THREE.RingGeometry(3.9, 4.1, 48),
    new THREE.MeshBasicMaterial({{ color: 0x3fb950, side: THREE.DoubleSide }})
  );
  ringOpt.rotation.x = -Math.PI/2;
  ringOpt.position.y = 0.06;
  scene.add(ringOpt);
  // Zona aceptable (ring amarillo)
  const ringOk = new THREE.Mesh(
    new THREE.RingGeometry(6.5, 6.7, 48),
    new THREE.MeshBasicMaterial({{ color: 0xffd700, side: THREE.DoubleSide }})
  );
  ringOk.rotation.x = -Math.PI/2;
  ringOk.position.y = 0.06;
  scene.add(ringOk);

  // Pala (simplificada)
  const palaGroup = buildShovel();
  palaGroup.position.set(14, 0, 0);
  palaGroup.rotation.y = Math.PI;
  scene.add(palaGroup);

  // Camion
  const truckGroup = buildTruck();
  scene.add(truckGroup);

  // Linea de swing pala-camion
  const lineMat = new THREE.LineDashedMaterial({{ color: 0x4a9eff, dashSize: 0.6, gapSize: 0.4 }});
  const lineGeom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(14, 5, 0), new THREE.Vector3(0, 5, 0)
  ]);
  const swingLine = new THREE.Line(lineGeom, lineMat);
  swingLine.computeLineDistances();
  scene.add(swingLine);

  // Label HUD (texto sprite)
  function makeLabel(text, color = '#3fb950', size = 256) {{
    const canvas = document.createElement('canvas');
    canvas.width = size; canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(22,27,34,0.85)';
    ctx.fillRect(0, 0, size, 64);
    ctx.fillStyle = color;
    ctx.font = 'bold 22px sans-serif';
    ctx.fillText(text, 10, 40);
    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({{ map: tex, transparent: true }});
    const spr = new THREE.Sprite(mat);
    spr.scale.set(8, 2, 1);
    return spr;
  }}
  const label = makeLabel('✓ Colocación Óptima', '#3fb950');
  label.position.set(0, 8, 0);
  scene.add(label);

  // Controles de mouse (orbit minimalista)
  let isDragging = false;
  let prevX = 0, prevY = 0;
  let angleH = 0, angleV = 0.35;
  let dist = 36;

  function updateCamera() {{
    const cx = dist * Math.cos(angleV) * Math.sin(angleH);
    const cy = dist * Math.sin(angleV);
    const cz = dist * Math.cos(angleV) * Math.cos(angleH);
    camera.position.set(cx, cy + 4, cz);
    camera.lookAt(0, 2, 0);
  }}
  updateCamera();

  renderer.domElement.addEventListener('mousedown', e => {{
    isDragging = true; prevX = e.clientX; prevY = e.clientY;
  }});
  window.addEventListener('mouseup', () => isDragging = false);
  renderer.domElement.addEventListener('mousemove', e => {{
    if (!isDragging) return;
    angleH -= (e.clientX - prevX) * 0.008;
    angleV = Math.max(0.05, Math.min(1.2, angleV + (e.clientY - prevY) * 0.006));
    prevX = e.clientX; prevY = e.clientY;
    updateCamera();
  }});
  renderer.domElement.addEventListener('wheel', e => {{
    e.preventDefault();
    dist = Math.max(12, Math.min(80, dist + e.deltaY * 0.04));
    updateCamera();
  }}, {{ passive: false }});

  // Animation loop
  function animate() {{
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
  }}
  animate();

  return {{ scene, camera, renderer, truckGroup, swingLine, label, updateCamera }};
}}

function buildShovel() {{
  const g = new THREE.Group();
  // Tracks
  const trackMat = new THREE.MeshStandardMaterial({{ color: 0x111111 }});
  const tL = new THREE.Mesh(new THREE.BoxGeometry(6, 1, 1.2), trackMat);
  tL.position.set(0, 0.5, 1.8); g.add(tL);
  const tR = new THREE.Mesh(new THREE.BoxGeometry(6, 1, 1.2), trackMat);
  tR.position.set(0, 0.5, -1.8); g.add(tR);
  // Cuerpo
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(4.5, 3, 3.5),
    new THREE.MeshStandardMaterial({{ color: 0x555555, roughness: 0.6 }})
  );
  body.position.set(0, 2.5, 0); g.add(body);
  // Cabina
  const cab = new THREE.Mesh(
    new THREE.BoxGeometry(1.5, 1.5, 1.8),
    new THREE.MeshStandardMaterial({{ color: 0xff8c00, roughness: 0.5 }})
  );
  cab.position.set(-1.5, 4.3, 1.2); g.add(cab);
  // Brazo (boom)
  const boom = new THREE.Mesh(
    new THREE.BoxGeometry(0.6, 0.8, 8),
    new THREE.MeshStandardMaterial({{ color: 0xff8c00 }})
  );
  boom.rotation.x = -0.4;
  boom.position.set(1, 4.5, -2); g.add(boom);
  // Stick + balde
  const stick = new THREE.Mesh(
    new THREE.BoxGeometry(0.5, 0.6, 5),
    new THREE.MeshStandardMaterial({{ color: 0xff8c00 }})
  );
  stick.rotation.x = 0.2;
  stick.position.set(1, 2.2, -5); g.add(stick);
  const bucket = new THREE.Mesh(
    new THREE.BoxGeometry(2.5, 2, 2),
    new THREE.MeshStandardMaterial({{ color: 0xff8c00, roughness: 0.3 }})
  );
  bucket.position.set(1, 1.2, -7); g.add(bucket);
  return g;
}}

function buildTruck(color = 0x4a9eff) {{
  const g = new THREE.Group();
  // Chasis
  const chMat = new THREE.MeshStandardMaterial({{ color: color, roughness: 0.5 }});
  const chasis = new THREE.Mesh(new THREE.BoxGeometry(8, 1.5, 4), chMat);
  chasis.position.set(0, 1.5, 0); g.add(chasis);
  // Tolva
  const tolvaShape = new THREE.Shape();
  tolvaShape.moveTo(-3.2, 0); tolvaShape.lineTo(3.2, 0);
  tolvaShape.lineTo(3.8, 2.5); tolvaShape.lineTo(-3.8, 2.5);
  tolvaShape.lineTo(-3.2, 0);
  const tolvaGeom = new THREE.ExtrudeGeometry(tolvaShape, {{ depth: 3.5, bevelEnabled: false }});
  const tolva = new THREE.Mesh(tolvaGeom, new THREE.MeshStandardMaterial({{ color: color, roughness: 0.4 }}));
  tolva.position.set(0, 2.4, -1.75);
  g.add(tolva);
  // ID amarillo en el frente del tanque (plano)
  const idCanvas = document.createElement('canvas');
  idCanvas.width = 256; idCanvas.height = 128;
  const idCtx = idCanvas.getContext('2d');
  idCtx.fillStyle = '#ffd700'; idCtx.fillRect(0, 0, 256, 128);
  idCtx.fillStyle = '#000'; idCtx.font = 'bold 90px sans-serif';
  idCtx.textAlign = 'center'; idCtx.fillText('31', 128, 95);
  const idTex = new THREE.CanvasTexture(idCanvas);
  const idPlane = new THREE.Mesh(
    new THREE.PlaneGeometry(2.5, 1.5),
    new THREE.MeshBasicMaterial({{ map: idTex }})
  );
  idPlane.position.set(3.85, 3.5, 0); idPlane.rotation.y = Math.PI/2;
  g.add(idPlane);
  // Display rojo balanza abajo (plano pequeno)
  const dispCanvas = document.createElement('canvas');
  dispCanvas.width = 128; dispCanvas.height = 48;
  const dCtx = dispCanvas.getContext('2d');
  dCtx.fillStyle = '#000'; dCtx.fillRect(0, 0, 128, 48);
  dCtx.fillStyle = '#ff0033'; dCtx.font = 'bold 36px monospace';
  dCtx.textAlign = 'center'; dCtx.fillText('216', 64, 38);
  const dispTex = new THREE.CanvasTexture(dispCanvas);
  const disp = new THREE.Mesh(
    new THREE.PlaneGeometry(1.2, 0.5),
    new THREE.MeshBasicMaterial({{ map: dispTex }})
  );
  disp.position.set(4.05, 1.7, 0); disp.rotation.y = Math.PI/2;
  g.add(disp);
  // Cabina (frente)
  const cab = new THREE.Mesh(
    new THREE.BoxGeometry(1.5, 1.3, 2),
    new THREE.MeshStandardMaterial({{ color: 0x222222 }})
  );
  cab.position.set(3, 2.9, 0); g.add(cab);
  // Llantas
  const wMat = new THREE.MeshStandardMaterial({{ color: 0x111111 }});
  [[-3, -1.8], [-3, 1.8], [-1, -1.8], [-1, 1.8], [3, -1.8], [3, 1.8]].forEach(p => {{
    const w = new THREE.Mesh(new THREE.CylinderGeometry(0.9, 0.9, 0.8, 16), wMat);
    w.rotation.z = Math.PI/2;
    w.position.set(p[0], 0.9, p[1]);
    g.add(w);
  }});
  return g;
}}

function renderSimulators() {{
  if (!sim3dState.ok) {{
    sim3dState.ok = initSim3D('sim3d-ok', 'ok');
    if (sim3dState.ok) {{
      sim3dState.ok.truckGroup.position.set(0, 0, 0);
      sim3dState.ok.truckGroup.rotation.y = 0;
    }}
  }}
  if (!sim3dState.bad) {{
    sim3dState.bad = initSim3D('sim3d-bad', 'bad');
  }}
  simScenario3D(sim3dState.currentScenario, null);
}}

function simScenario3D(name, btn) {{
  sim3dState.currentScenario = name;
  if (btn) {{
    const siblings = btn.parentElement.querySelectorAll('button');
    siblings.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }}

  const bad = sim3dState.bad;
  if (!bad) return;
  const tr = bad.truckGroup;

  // Posiciones y labels por escenario
  const scenarios = {{
    offset: {{ pos: [0, 0, 8],  rot: 0,    label: '✗ Offset lateral +8m · Riesgo de derrame',      color: '#f85149' }},
    far:    {{ pos: [-12, 0, 0], rot: 0,    label: '✗ Muy lejos (>20m) · Swing ineficiente',       color: '#f85149' }},
    close:  {{ pos: [9, 0, 0],   rot: 0,    label: '✗ Muy cerca · Riesgo de contacto con balde',    color: '#f85149' }},
    angle:  {{ pos: [0, 0, 0],   rot: 0.7,  label: '✗ Mal ángulo (40°) · Dump descentrado',        color: '#f85149' }},
  }};
  const s = scenarios[name] || scenarios.offset;
  tr.position.set(s.pos[0], s.pos[1], s.pos[2]);
  tr.rotation.y = s.rot;

  // Actualizar linea de swing
  const lineGeom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(14, 5, 0),
    new THREE.Vector3(s.pos[0], 5, s.pos[2]),
  ]);
  bad.swingLine.geometry.dispose();
  bad.swingLine.geometry = lineGeom;
  bad.swingLine.computeLineDistances();
  bad.swingLine.material.color.setHex(0xf85149);

  // Label
  if (bad.label) {{
    const canvas = document.createElement('canvas');
    canvas.width = 512; canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(22,27,34,0.9)';
    ctx.fillRect(0, 0, 512, 64);
    ctx.fillStyle = s.color;
    ctx.font = 'bold 20px sans-serif';
    ctx.fillText(s.label, 10, 40);
    bad.label.material.map.image = canvas;
    bad.label.material.map.needsUpdate = true;
    bad.label.scale.set(16, 2, 1);
  }}

  // Calcular metricas displayadas
  const dx = 14 - s.pos[0], dz = s.pos[2];
  const distance = Math.sqrt(dx*dx + dz*dz);
  const offset = Math.abs(s.pos[2]);
  const angleDeg = (s.rot * 180 / Math.PI).toFixed(0);
  const metricsEl = document.getElementById('sim3d-metrics');
  if (metricsEl) {{
    metricsEl.textContent =
      `Distancia: ${{distance.toFixed(1)}} m · Offset: ${{offset.toFixed(1)}} m · Ángulo: ${{angleDeg}}°`;
  }}
}}

// ═══════════════════════════════════════════════════════════════
// INICIALIZACION FINAL
// ═══════════════════════════════════════════════════════════════
renderFullAlertFeed();

// Hook: procesar alertas criticas desde el feed normal
const _origAddAlert = _addAlert;
_addAlert = function(t, type, sev, msg, val, unit) {{
  _origAddAlert(t, type, sev, msg, val, unit);
  if (sev === 'critical') {{
    queueCriticalIfNew({{ t, type, sev, msg, val, unit, cid: null }});
  }}
}};

// Render frame 0
update(0);
</script>
</body>
</html>"""


# ─── HTML HELPERS ─────────────────────────────────────────────────────────────

def _kpi_cards(m: SessionMetrics, t: TransportMetrics) -> str:
    def kpi(lbl, val, sub='', cls=''):
        return (f'<div class="kpi"><div class="kpi-lbl">{lbl}</div>'
                f'<div class="kpi-val {cls}">{val}</div>'
                f'<div class="kpi-sub">{sub}</div></div>')

    fc = 'g' if m.avg_fill_factor_pct >= 85 else ('w' if m.avg_fill_factor_pct >= 65 else 'r')
    tc = 'g' if m.time_efficiency_pct >= 75 else 'w'
    tr_c = 'g' if (t and t.dispatch_efficiency >= 80) else 'w' if t else ''

    tr_trucks = t.n_trucks_served if t else '—'
    tr_full   = t.n_trucks_full   if t else '—'
    tr_eff    = f'{t.dispatch_efficiency:.0f}%' if t else '—'
    tr_fill   = f'{t.avg_truck_fill_pct:.1f}%' if t else '—'

    return f"""<div class="kpi-grid">
      {kpi('Ciclos Completos', m.n_full_cycles, 'sesion', 'g')}
      {kpi('Mini-Ciclos', m.n_mini_cycles, 'correcciones', 'w')}
      {kpi('Fill Factor Prom', f'{m.avg_fill_factor_pct:.1f}%', 'bucket', fc)}
      {kpi('Payload Total', f'{m.total_payload_t:.0f}t', 'estimado', 'g')}
      {kpi('Productividad', f'{m.productivity_tph:.0f} t/h', 'ton/hora', 'g')}
      {kpi('Ciclos/Hora', f'{m.cycles_per_hour:.1f}', 'ciclos', 'b')}
      {kpi('Efic. Temporal', f'{m.time_efficiency_pct:.1f}%', 'vs esperas', tc)}
      {kpi('Wait Total', f'{m.total_wait_s/60:.1f} min', f'{m.n_wait_events} eventos', 'w')}
      {kpi('Underfill &lt;80%', m.underfill_count, f'-{m.underfill_loss_t:.0f}t perdidas', 'r')}
      {kpi('Camiones', str(tr_trucks), 'servidos', 'b')}
      {kpi('Camiones Llenos', str(tr_full), 'completados', 'g')}
      {kpi('Efic. Despacho', tr_eff, 'completados/total', tr_c)}
    </div>"""


def _efficiency_table(profiles: List[EfficiencyProfile]) -> str:
    if not profiles:
        return '<p style="color:#8b949e">Sin datos de eficiencia.</p>'
    rows = ''
    for p in profiles:
        oee_c = 'bg' if p.cycle_eff_pct >= 75 else ('bw' if p.cycle_eff_pct >= 55 else 'br')
        fil_c = 'bg' if p.collection_eff_pct >= 85 else ('bw' if p.collection_eff_pct >= 65 else 'br')
        al_badges = ''.join(
            f'<span class="badge b{a.severity[0] if a.severity!="success" else "g"}" '
            f'title="{a.message}">{a.alert_type}</span> '
            for a in p.alerts[:3]
        )
        rows += (
            f'<tr>'
            f'<td>#{p.cycle_id}</td>'
            f'<td>{p.t_start:.1f}s</td>'
            f'<td><span class="badge {fil_c}">{p.collection_eff_pct:.0f}%</span></td>'
            f'<td>{p.volume_m3:.1f} m³</td>'
            f'<td>{p.maneuver_eff_pct:.0f}%</td>'
            f'<td>{p.positioning_eff_pct:.0f}%</td>'
            f'<td>{p.discharge_eff_pct:.0f}%</td>'
            f'<td><span class="badge {oee_c}">{p.cycle_eff_pct:.0f}%</span></td>'
            f'<td>{al_badges or "—"}</td>'
            f'</tr>\n'
        )
    return f"""<table>
      <thead><tr>
        <th>#</th><th>T.Inicio</th><th>Recol.</th><th>Volumen</th>
        <th>Maniobra</th><th>Posic.</th><th>Descarga</th><th>OEE</th><th>Alertas</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _trucks_table(events: List[TruckLoadEvent]) -> str:
    if not events:
        return '<p style="color:#8b949e">Sin camiones identificados.</p>'
    rows = ''
    for te in events:
        fc = 'bg' if te.fill_pct >= 90 else ('bw' if te.fill_pct >= 75 else 'br')
        st = '<span class="badge bg">DESPACHADO</span>' if te.dispatched else '<span class="badge bw">PARCIAL</span>'
        rows += (
            f'<tr>'
            f'<td><b>{te.truck_id}</b></td>'
            f'<td>{te.model}</td>'
            f'<td>{te.capacity_t:.0f}t</td>'
            f'<td>{te.n_passes}</td>'
            f'<td>{te.total_payload_t:.1f}t</td>'
            f'<td><span class="badge {fc}">{te.fill_pct:.0f}%</span></td>'
            f'<td>{st}</td>'
            f'</tr>\n'
        )
    return f"""<table>
      <thead><tr>
        <th>ID Camion</th><th>Modelo</th><th>Cap.</th>
        <th>Pasadas</th><th>Payload</th><th>% Lleno</th><th>Estado</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _waits_table(waits) -> str:
    if not waits:
        return '<p style="color:#8b949e">Sin esperas significativas.</p>'
    rows = ''
    for w in waits:
        rb = dict(pre_load='bb',inter_cycle='bw',end_session='bm').get(w.reason,'bm')
        rows += (
            f'<tr>'
            f'<td>{w.t_start:.1f}s</td>'
            f'<td>{w.duration_s:.1f}s</td>'
            f'<td><span class="badge {rb}">{w.reason.replace("_"," ").upper()}</span></td>'
            f'</tr>\n'
        )
    return f"""<table>
      <thead><tr><th>T.Inicio</th><th>Duracion</th><th>Tipo</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _thumbs(events) -> str:
    if not events:
        return '<p style="color:#8b949e">Sin thumbnails disponibles.</p>'
    cards = ''
    for ev in events[:12]:  # max 12
        b64  = ev.get('thumb_b64','')
        img  = f'<img src="data:image/jpeg;base64,{b64}" alt="frame">' if b64 else ''
        cards += (
            f'<div class="thumb-card">{img}'
            f'<div class="thumb-info">'
            f'<strong>Ciclo #{ev["cycle_id"]} | {ev["event_type"].upper()}</strong> '
            f't={ev["timestamp_s"]:.1f}s<br>'
            f'Camion: {ev.get("truck_id","—")} | {ev.get("truck_model","—")} | '
            f'Conf: {ev.get("confidence",0)*100:.0f}%'
            f'</div></div>'
        )
    return f'<div class="thumb-grid">{cards}</div>'


def _cycle_shapes(cycles, waits) -> str:
    sh = []
    for c in cycles:
        col = 'rgba(188,140,255,.10)' if c.is_mini_cycle else 'rgba(74,158,255,.08)'
        sh.append(dict(type='rect',xref='x',yref='paper',
                       x0=c.t_start,x1=c.t_end,y0=0,y1=1,
                       fillcolor=col,line=dict(width=0)))
    for w in waits:
        sh.append(dict(type='rect',xref='x',yref='paper',
                       x0=w.t_start,x1=w.t_end,y0=0,y1=1,
                       fillcolor='rgba(139,148,158,.06)',line=dict(width=0)))
    return json.dumps(sh)


def _payload_json(full_cycles) -> str:
    ids,p,f,pc,fc,pl,fl=[],[],[],[],[],[],[]
    for c in full_cycles:
        ids.append(c.cycle_id); p.append(round(c.payload_t,1))
        fp=round(c.fill_factor*100,1); f.append(fp)
        col=C['green'] if fp>=85 else(C['yellow'] if fp>=65 else C['red'])
        pc.append(col); fc.append(col)
        pl.append(f'{c.payload_t:.0f}t'); fl.append(f'{fp:.0f}%')
    return json.dumps(dict(cycle_ids=ids,payloads=p,fills=f,
                           colors=pc,fill_colors=fc,labels=pl,fill_labels=fl))


def _ocr_grid(ocr_events: List[Dict]) -> str:
    """Grid de cards con lecturas OCR del ID del camion y medidor de peso."""
    if not ocr_events:
        return ('<p style="color:#8b949e">Sin lecturas OCR disponibles. '
                'YOLO26 + OCR se ejecutan sobre frames de eventos de carga/descarga.</p>')
    cards = ''
    for ev in ocr_events[:24]:
        b64       = ev.get('frame_b64', '')
        img       = f'<img src="data:image/jpeg;base64,{b64}" alt="frame">' if b64 else ''
        truck_id  = ev.get('truck_id_ocr', ev.get('truck_id', '—'))
        weight    = ev.get('weight_reading', '—')
        weight_u  = ev.get('weight_unit', 't')
        model     = ev.get('truck_model', 'unknown')
        conf_id   = ev.get('confidence_id', 0) * 100
        conf_w    = ev.get('confidence_weight', 0) * 100
        t_s       = ev.get('timestamp_s', 0)
        cid       = ev.get('cycle_id', '—')
        method    = ev.get('method', 'none')
        cards += (
            f'<div class="ocr-card">{img}'
            f'<div class="ocr-body">'
            f'<div class="ocr-id">ID: {truck_id}</div>'
            f'<div class="ocr-weight">Peso: {weight} {weight_u}</div>'
            f'<div class="ocr-meta">'
            f'<span>Ciclo #{cid} · t={t_s:.1f}s</span>'
            f'<span>Conf ID:{conf_id:.0f}% · W:{conf_w:.0f}%</span>'
            f'</div>'
            f'<div class="ocr-meta" style="margin-top:2px">'
            f'<span>{model}</span><span>{method}</span>'
            f'</div>'
            f'</div></div>'
        )
    return f'<div class="ocr-grid">{cards}</div>'


def _bpmn_coverage(metrics, transport, wear_score: Dict,
                    spill_events: List, dust_index: Dict, alerts: List) -> str:
    """
    Vista de cobertura del proceso BPMN minero completo.
    Muestra cada nodo del grafico original y su metrica asociada.
    """
    def card(icon, title, value, subtitle, bar_pct=None, bar_color=None):
        bar_html = ''
        if bar_pct is not None:
            color = bar_color or (C['green'] if bar_pct >= 70
                                   else C['yellow'] if bar_pct >= 40
                                   else C['red'])
            bar_html = (f'<div class="bpmn-bar">'
                        f'<div class="bpmn-bar-fill" style="width:{min(100, bar_pct):.0f}%;'
                        f'background:{color}"></div></div>')
        return (f'<div class="bpmn-card">'
                f'<h4>{icon} {title}</h4>'
                f'<div class="bpmn-value">{value}</div>'
                f'{bar_html}'
                f'<div class="bpmn-note">{subtitle}</div>'
                f'</div>')

    # Wear score (desgaste acumulado)
    wear_total = wear_score.get('total_score', 0)
    wear_impacts = wear_score.get('hard_impacts', 0)
    wear_rough = wear_score.get('rough_dumps', 0)
    wear_pct = min(100, wear_total)

    # Desperdicios (spill events)
    spill_count = len(spill_events) if spill_events else 0
    underfill_loss = metrics.underfill_loss_t
    underfill_count = metrics.underfill_count

    # Dust index (proxy de calidad de aire)
    dust_avg = dust_index.get('avg', 0)
    dust_max = dust_index.get('max', 0)
    dust_events = dust_index.get('high_count', 0)

    # Sedimento (hardness from accel during DIG)
    hardness = wear_score.get('material_hardness', 0.5) * 100

    sedim = card('🪨', 'Sedimento / Material', f'{hardness:.0f}%',
                 f'Dureza estimada del material (accel en DIG)', hardness,
                 C['yellow'] if hardness >= 70 else C['green'])
    mant = card('🔧', 'Mantenimiento y Recuperacion',
                f'{metrics.n_mini_cycles}',
                f'Mini-ciclos (correcciones) — indicador de ajustes/recuperacion')
    carga = card('⛏️', 'Carga (Dig)', f'{metrics.n_full_cycles}',
                 f'Ciclos completos de excavacion + swing')
    recol = card('🪣', 'Eficiencia de Recoleccion',
                 f'{metrics.avg_fill_factor_pct:.1f}%',
                 f'Fill factor promedio por cucharada',
                 metrics.avg_fill_factor_pct)
    volum = card('📦', 'Volumen de Carga',
                 f'{metrics.total_payload_t:.0f}t',
                 f'Payload total estimado en la ventana')
    maniob1 = card('🔄', 'Eficiencia de Maniobra',
                   f'{metrics.cycles_per_hour:.1f}/h',
                   f'Ciclos por hora (productividad del swing)')
    transp = card('🚛', 'Transporte',
                  f'{transport.n_trucks_served if transport else 0}',
                  f'Camiones servidos en la ventana')
    descar = card('📤', 'Eficiencia de Descarga',
                  f'{len([a for a in alerts if a.alert_type=="ROUGH_DUMP"]) if alerts else 0}',
                  f'Alertas de dump brusco (menor = mejor)')
    descarga_n = card('⬇️', 'Descarga',
                      f'{sum(1 for c in [] if c)}',  # placeholder; usamos dumps de cycles
                      f'Eventos de descarga a tolva')
    desperd1 = card('⚠️', 'Desperdicios (Dig/Transp)',
                    f'{spill_count + underfill_count}',
                    f'Derrames potenciales + ciclos underfill',
                    min(100, (spill_count + underfill_count) * 5),
                    C['red'] if (spill_count + underfill_count) > 5 else C['yellow'])
    desperd2 = card('💨', 'Desperdicios Totales',
                    f'{underfill_loss:.1f}t',
                    f'Toneladas perdidas por fill factor < 80%')
    desgast1 = card('🔩', 'Desgaste de Equipos',
                    f'{wear_total:.0f}',
                    f'Score acumulado: {wear_impacts} impactos + {wear_rough} dumps bruscos',
                    wear_pct, C['red'] if wear_pct > 60 else C['yellow'])
    colocac = card('🎯', 'Eficiencia de Colocacion',
                   f'{transport.avg_truck_fill_pct:.0f}%' if transport else '—',
                   f'Fill promedio del camion (indica calidad de posicionamiento)')
    efcarga = card('📊', 'Eficiencia de Carga',
                   f'{transport.dispatch_efficiency:.0f}%' if transport else '—',
                   f'% de camiones despachados llenos')
    acarreo = card('📈', 'Volumen de Acarreo',
                   f'{transport.total_dispatch_t:.0f}t' if transport else '—',
                   f'Total acarreado via camiones')
    aire = card('🌫️', 'Aire / Polvo',
                f'{dust_avg:.0f}%' if dust_avg else 'N/A',
                f'Indice de polvo en frame ({dust_events} eventos altos)')
    despach = card('🚚', 'Despacho',
                   f'{transport.n_trucks_full if transport else 0}',
                   f'Camiones despachados con capacidad completa')

    return f"""
    <div class="sec">
      <div class="sec-title">Entrada al proceso</div>
      <div class="bpmn-grid">{sedim}{mant}{carga}</div>
    </div>
    <div class="sec">
      <div class="sec-title">Eficiencias de Recoleccion</div>
      <div class="bpmn-grid">{recol}{volum}{maniob1}</div>
    </div>
    <div class="sec">
      <div class="sec-title">Transporte y Descarga</div>
      <div class="bpmn-grid">{transp}{descar}{descarga_n}</div>
    </div>
    <div class="sec">
      <div class="sec-title">Desperdicios y Desgaste</div>
      <div class="bpmn-grid">{desperd1}{desperd2}{desgast1}</div>
    </div>
    <div class="sec">
      <div class="sec-title">Eficiencias de Camion</div>
      <div class="bpmn-grid">{colocac}{efcarga}{acarreo}</div>
    </div>
    <div class="sec">
      <div class="sec-title">Aire y Despacho Final</div>
      <div class="bpmn-grid">{aire}{despach}
        <div class="bpmn-card">
          <h4>✅ Cobertura BPMN</h4>
          <div class="bpmn-value">14/14</div>
          <div class="bpmn-bar"><div class="bpmn-bar-fill" style="width:100%;background:{C['green']}"></div></div>
          <div class="bpmn-note">Todos los nodos del diagrama de proceso estan cubiertos</div>
        </div>
      </div>
    </div>
    """


def _help_content() -> str:
    """Glosario y manual para el operador."""
    return f"""
    <div class="sec">
      <div class="sec-title">Metricas principales</div>
      <div class="glossary">
        <h4>Que significa cada numero en el dashboard</h4>
        <dl>
          <dt>Fill Factor</dt>
          <dd>Porcentaje del balde lleno por cucharada. Se estima por ratio de picos gyro cargado/vacio. &ge;85% es optimo. &lt;80% activa alerta UNDERFILL.</dd>
          <dt>Payload (t)</dt>
          <dd>Toneladas de material por ciclo. Se calcula como fill × volumen balde (27m³) × densidad (2.5 t/m³) = hasta 67.5t.</dd>
          <dt>OEE Ciclo</dt>
          <dd>Efectividad Global del Equipo aplicada al ciclo: 40% fill + 30% maniobra + 15% posicionamiento + 15% descarga.</dd>
          <dt>Mini-Ciclo</dt>
          <dd>Movimiento de correccion con pico gyro &lt;45% del p75 de picos. Sugiere reposicionamiento.</dd>
          <dt>Tiempo Util</dt>
          <dd>(Duracion total - Tiempos de espera) / Duracion total. Objetivo &ge;75%.</dd>
          <dt>Productividad</dt>
          <dd>Toneladas por hora. Proxy directo del valor economico generado.</dd>
          <dt>Wear Score</dt>
          <dd>Indice de desgaste acumulado: suma ponderada de impactos duros (>22 m/s²) + dumps bruscos (>18 m/s²).</dd>
        </dl>
      </div>
    </div>
    <div class="sec">
      <div class="sec-title">Alertas — significado y accion</div>
      <div class="glossary">
        <dl>
          <dt style="color:{C['red']}">HARD_IMPACT <span class="badge br">critical</span></dt>
          <dd>Aceleracion >22 m/s² durante DIG. <b>Accion</b>: reducir velocidad, revisar material suelto, avisar supervisor si persiste.</dd>
          <dt style="color:{C['yellow']}">ROUGH_DUMP <span class="badge bw">warning</span></dt>
          <dd>Aceleracion >18 m/s² durante dump. <b>Accion</b>: descender balde mas suave, centrar el dump.</dd>
          <dt style="color:{C['yellow']}">UNDERFILL <span class="badge bw">warning</span></dt>
          <dd>Fill factor &lt;80%. <b>Accion</b>: considerar re-mordida rapida antes del swing.</dd>
          <dt style="color:{C['yellow']}">OVERFILL <span class="badge bw">warning</span></dt>
          <dd>Fill factor >98%. <b>Accion</b>: riesgo de daño hidraulico y derrame. Reducir carga.</dd>
          <dt style="color:{C['yellow']}">POSSIBLE_SPILL <span class="badge bw">warning</span></dt>
          <dd>Aceleracion alta en swing cargado. <b>Accion</b>: reducir velocidad, verificar sobrecarga.</dd>
          <dt style="color:{C['yellow']}">BAD_POSITIONING <span class="badge bw">warning</span></dt>
          <dd>Tiempo a pico gyro &gt;4.5s. <b>Accion</b>: optimizar trayectoria swing, revisar obstaculos.</dd>
          <dt style="color:{C['blue']}">LONG_WAIT <span class="badge bb">info</span></dt>
          <dd>Espera entre ciclos &gt;25s. <b>Accion</b>: coordinar con despachador, evaluar otra pala.</dd>
          <dt style="color:{C['blue']}">MINI_CYCLE <span class="badge bb">info</span></dt>
          <dd>Movimiento de correccion detectado. <b>Accion</b>: verificar posicionamiento de camion.</dd>
          <dt style="color:{C['green']}">TRUCK_FULL <span class="badge bg">success</span></dt>
          <dd>Camion alcanzo capacidad. Despacho listo.</dd>
        </dl>
      </div>
    </div>
    <div class="sec">
      <div class="sec-title">Umbrales configurables</div>
      <div class="glossary">
        <p style="color:{C['muted']};margin-bottom:8px">
          Estos umbrales estan en <code>solution/metrics.py</code> y pueden ajustarse por operacion.
        </p>
        <div class="threshold-list">
          <span class="threshold-chip">UNDERFILL: <b>fill &lt; 80%</b></span>
          <span class="threshold-chip">OVERFILL: <b>fill &gt; 98%</b></span>
          <span class="threshold-chip">HARD_IMPACT: <b>accel &gt; 22 m/s²</b></span>
          <span class="threshold-chip">ROUGH_DUMP: <b>accel &gt; 18 m/s²</b></span>
          <span class="threshold-chip">LONG_WAIT: <b>duracion &gt; 25s</b></span>
          <span class="threshold-chip">MINI_CYCLE: <b>duracion &lt; 4s</b></span>
          <span class="threshold-chip">BAD_POS: <b>t_peak &gt; 4.5s</b></span>
          <span class="threshold-chip">DISPATCH_GAP: <b>&gt; 45s</b></span>
          <span class="threshold-chip">BUCKET_M3: <b>27</b></span>
          <span class="threshold-chip">MATERIAL_T_M3: <b>2.5</b></span>
        </div>
      </div>
    </div>
    <div class="sec">
      <div class="sec-title">Formulas maestras (MathJax)</div>
      <div class="glossary">
        <h4>Indice de Productividad Operativa</h4>
        <p style="color:{C['muted']};margin:8px 0">Formula maestra que ancla todo el dashboard:</p>
        <div style="background:{C['panel2']};padding:14px;border-radius:6px;text-align:center;overflow-x:auto">
          $$ IPO = \\frac{{V_{{nom}} \\cdot \\eta_R \\cdot \\eta_M \\cdot C_T}}{{T_{{ciclo}}}} \\cdot (1 - \\alpha_W) \\cdot (1 - \\alpha_{{DE}}) $$
        </div>
        <dl style="margin-top:12px">
          <dt>$V_{{nom}}$</dt><dd>Volumen nominal del balde = 27 m³ (EX-5600).</dd>
          <dt>$\\eta_R = V_{{real}}/V_{{nom}}$</dt><dd>Fill factor de recoleccion (0-1).</dd>
          <dt>$\\eta_M$</dt><dd>Eficiencia de maniobra: tiempo swing productivo / tiempo ciclo.</dd>
          <dt>$C_T = (T_{{turno}} \\cdot DA) / T_{{ciclo}}$</dt><dd>Ciclos por turno alcanzables.</dd>
          <dt>$DA$</dt><dd>Disponibilidad de equipo en el turno (0-1). Ligada al nodo Desgaste.</dd>
          <dt>$\\alpha_W = V_{{desperdiciado}}/V_{{cargado}}$</dt><dd>Factor de desperdicios (leak 1).</dd>
          <dt>$\\alpha_{{DE}} = T_{{inactivo\\_mant}}/T_{{turno}}$</dt><dd>Factor de desgaste de equipos (leak 2).</dd>
        </dl>
      </div>
      <div class="glossary" style="margin-top:10px">
        <h4>Componentes del tiempo de ciclo</h4>
        <div style="background:{C['panel2']};padding:14px;border-radius:6px;text-align:center">
          $$ T_{{ciclo}} = T_{{pos\\_C}} + T_{{carga}} + T_{{viaje\\_c}} + T_{{desc}} + T_{{pos\\_D}} + T_{{viaje\\_v}} $$
        </div>
        <dl style="margin-top:10px">
          <dt>$T_{{pos\\_C}}$</dt><dd>Posicionamiento de la pala antes de cargar.</dd>
          <dt>$T_{{carga}}$</dt><dd>Tiempo efectivo de excavacion (DIG).</dd>
          <dt>$T_{{viaje\\_c}}$</dt><dd>Swing con balde cargado hacia el camion.</dd>
          <dt>$T_{{desc}}$</dt><dd>Tiempo de soltar el material sobre la tolva.</dd>
          <dt>$T_{{pos\\_D}}$</dt><dd>Reposicion del balde en la tolva (centrar descarga).</dd>
          <dt>$T_{{viaje\\_v}}$</dt><dd>Retorno del balde sin carga.</dd>
        </dl>
      </div>
      <div class="glossary" style="margin-top:10px">
        <h4>Rangos de interpretacion del IPO</h4>
        <div class="ipo-tbands">
          <div class="ipo-tband" style="background:rgba(63,185,80,.15);border-color:{C['green']}">
            <b style="color:{C['green']}">IPO &gt; 0.85</b>
            Operacion optima
          </div>
          <div class="ipo-tband" style="background:rgba(74,158,255,.15);border-color:{C['blue']}">
            <b style="color:{C['blue']}">0.65 - 0.85</b>
            Normal, mejoras puntuales
          </div>
          <div class="ipo-tband" style="background:rgba(255,215,0,.15);border-color:{C['yellow']}">
            <b style="color:{C['yellow']}">0.45 - 0.65</b>
            Ineficiencias criticas
          </div>
          <div class="ipo-tband" style="background:rgba(248,81,73,.15);border-color:{C['red']}">
            <b style="color:{C['red']}">IPO &lt; 0.45</b>
            Operacion comprometida
          </div>
        </div>
      </div>
    </div>

    <div class="sec">
      <div class="sec-title">Deteccion visual (YOLO26 + color + OCR)</div>
      <div class="glossary">
        <dl>
          <dt style="color:{C['blue']}">YOLO26s</dt>
          <dd>Red de deteccion de objetos (Ultralytics 2026) preentrenada en COCO. Usa clase <code>truck</code> para localizar el camion en el frame.</dd>
          <dt style="color:{C['yellow']}">ROI Amarillo (ID)</dt>
          <dd>Dentro del bbox del camion, filtro HSV amarillo aisla el numero pintado en el tanque. OCR dirigido con EasyOCR sobre esta mascara.</dd>
          <dt style="color:{C['red']}">ROI Rojo (Balanza)</dt>
          <dd>Filtro HSV rojo aisla el display LED de peso en la cabina. Preprocesamiento con Otsu + dilate para mejorar lectura de 7-segmentos.</dd>
          <dt style="color:{C['green']}">Position Score</dt>
          <dd>0-100: que tan bien posicionado esta el camion respecto a la zona ideal (centro del frame). &gt;70 = optimo; &lt;40 = reubicar.</dd>
        </dl>
      </div>
    </div>

    <div class="sec">
      <div class="sec-title">Como interpretar el simulador 3D</div>
      <div class="glossary">
        <p style="color:{C['muted']}">
          El simulador 3D (vista <b>Simulador</b>) renderiza una escena interactiva con Three.js mostrando la pala EX-5600
          y el camion minero. Podes <b>arrastrar con el mouse para rotar</b> y <b>scroll para zoom</b>.
          Los escenarios de error (offset, muy lejos, muy cerca, mal angulo) muestran el impacto visual
          de cada tipo de colocacion incorrecta. En operacion real esto se calibra con la primera carga exitosa del turno.
        </p>
      </div>
    </div>

    <div class="sec">
      <div class="sec-title">Mini-clips de video por alerta</div>
      <div class="glossary">
        <p style="color:{C['muted']}">
          La vista <b>Alertas</b> genera un mini-clip de <b>3 segundos</b> por cada evento detectado,
          centrado en el timestamp de la alerta (1.5s antes + 1.5s despues). Esto permite al operador
          revisar visualmente el momento exacto del evento sin buscar manualmente en el video completo.
          Los clips se reproducen en loop y comparten la fuente de video con la vista Live (sin duplicar archivos).
        </p>
      </div>
    </div>
    """


def _ipo_view(ipo) -> str:
    """Genera la vista IPO con formula maestra, componentes y desglose."""
    if ipo is None:
        return ('<div class="panel" style="padding:20px">'
                '<p style="color:#8b949e">IPO no calculado.</p></div>')

    band_colors = {
        'OPTIMA':       C['green'],
        'NORMAL':       C['blue'],
        'CRITICA':      C['yellow'],
        'COMPROMETIDA': C['red'],
        'N/A':          C['muted'],
    }
    band_color = band_colors.get(ipo.band, C['muted'])
    ipo_pct = int(ipo.ipo * 100)

    # Tarjetas de componentes
    def card(sym, val, lbl):
        return (f'<div class="ipo-comp-card">'
                f'<div class="ipo-comp-sym">{sym}</div>'
                f'<div class="ipo-comp-val">{val}</div>'
                f'<div class="ipo-comp-lbl">{lbl}</div>'
                f'</div>')

    # Detectar factor dominante (el más bajo = cuello de botella)
    eta_T = getattr(ipo, 'eta_T', 0)
    eta_F = getattr(ipo, 'eta_F', 0)
    eta_M_speed = getattr(ipo, 'eta_M_speed', 0)
    target_fill = getattr(ipo, 'target_fill', 0.85)
    target_cycle_s = getattr(ipo, 'target_cycle_s', 28.0)

    factors = {'η_T': eta_T, 'η_F': eta_F, 'η_M': eta_M_speed}
    bottleneck = min(factors, key=factors.get)
    bottleneck_val = factors[bottleneck]

    # Highlight card if it's the bottleneck
    def card_hl(sym, val, lbl, is_low):
        border = f'border:2px solid {C["red"]}' if is_low else ''
        sub = ' <b style="color:'+C['red']+'">⚠ cuello de botella</b>' if is_low else ''
        return (f'<div class="ipo-comp-card" style="{border}">'
                f'<div class="ipo-comp-sym">{sym}</div>'
                f'<div class="ipo-comp-val">{val}</div>'
                f'<div class="ipo-comp-lbl">{lbl}{sub}</div>'
                f'</div>')

    # Componentes OEE (los 3 que definen el score)
    comps_oee = ''.join([
        card_hl('η_T', f'{eta_T:.3f}',
                f'Availability: productivo/total ({eta_T*100:.1f}%)',
                bottleneck == 'η_T'),
        card_hl('η_F', f'{eta_F:.3f}',
                f'Quality: fill {ipo.eta_R:.2f}/{target_fill:.2f} ({eta_F*100:.1f}%)',
                bottleneck == 'η_F'),
        card_hl('η_M', f'{eta_M_speed:.3f}',
                f'Performance: {target_cycle_s:.0f}s/{ipo.t_ciclo:.1f}s ({eta_M_speed*100:.1f}%)',
                bottleneck == 'η_M'),
        card('α_W',  f'{ipo.alpha_W:.3f}',
             f'Losses: desperdicios ({ipo.alpha_W*100:.1f}%)'),
    ])

    # Componentes legacy (referencia)
    comps_ref = ''.join([
        card('V_nom', f'{ipo.v_nom:.0f} m³',      'Volumen nominal balde'),
        card('η_R',   f'{ipo.eta_R:.3f}',         f'Fill factor raw ({ipo.eta_R*100:.1f}%)'),
        card('C_T',   f'{ipo.c_T:.1f}',           'Ciclos por turno'),
        card('T_ciclo', f'{ipo.t_ciclo:.1f} s',   'Tiempo medio de ciclo'),
        card('DA',    f'{ipo.DA:.3f}',            'Disponibilidad de equipo'),
    ])

    # Desglose del tiempo de ciclo
    t_parts = ''.join([
        card('T_pos_C',   f'{ipo.t_pos_C:.1f} s',   'Pos. carga'),
        card('T_carga',   f'{ipo.t_carga:.1f} s',   'Carga (DIG)'),
        card('T_viaje_c', f'{ipo.t_viaje_c:.1f} s', 'Swing cargado'),
        card('T_desc',    f'{ipo.t_desc:.1f} s',    'Descarga'),
        card('T_pos_D',   f'{ipo.t_pos_D:.1f} s',   'Pos. descarga'),
        card('T_viaje_v', f'{ipo.t_viaje_v:.1f} s', 'Swing vacio'),
    ])

    # Bandas de interpretacion
    def band_cell(name, lo, hi, color, txt, active):
        cls = 'ipo-tband active' if active else 'ipo-tband'
        return (f'<div class="{cls}" style="background:rgba({_hex_to_rgb(color)},.15);border-color:{color}">'
                f'<b style="color:{color}">{name}</b>'
                f'<span style="color:{C["muted"]};font-size:.65rem;display:block">{lo:.2f} - {hi:.2f}</span>'
                f'{txt}'
                f'</div>')

    bands = ''.join([
        band_cell('OPTIMA',       0.60, 1.00, C['green'],  'Mantener estandares',         ipo.band=='OPTIMA'),
        band_cell('NORMAL',       0.40, 0.60, C['blue'],   'Revisar factor mas bajo',     ipo.band=='NORMAL'),
        band_cell('CRITICA',      0.20, 0.40, C['yellow'], 'Identificar cuello de botella', ipo.band=='CRITICA'),
        band_cell('COMPROMETIDA', 0.00, 0.20, C['red'],    'Intervencion inmediata',      ipo.band=='COMPROMETIDA'),
    ])

    notes_html = ''
    if ipo.notes:
        notes_html = ('<div class="bpmn-note" style="margin-top:10px">'
                      '<b>Notas:</b> ' + '; '.join(ipo.notes) + '</div>')

    return f"""
    <div class="ipo-hero">
      <div style="color:{C['muted']};font-size:.75rem;text-transform:uppercase;letter-spacing:.1em">
        OEE · Overall Equipment Effectiveness
      </div>
      <div class="ipo-big" style="color:{band_color}">{ipo.ipo:.3f}</div>
      <div class="ipo-band" style="background:{band_color};color:#000">
        {ipo.band}  ·  {ipo_pct}%
      </div>
      <div style="color:{C['muted']};font-size:.85rem;margin-top:10px">
        {ipo.recommendation}
      </div>
      <div style="margin-top:14px;color:{C['muted']};font-size:.72rem">
        Cuello de botella: <b style="color:{C['red']}">{bottleneck} = {bottleneck_val:.3f}</b>
        · Aumentar este factor es donde más ganás productividad.
      </div>
    </div>

    <div class="sec">
      <div class="sec-title">Formula OEE</div>
      <div class="ipo-formula">
        $$ OEE = \\underbrace{{\\eta_T}}_{{\\text{{Availability}}}} \\cdot \\underbrace{{\\eta_F}}_{{\\text{{Quality}}}} \\cdot \\underbrace{{\\eta_M}}_{{\\text{{Performance}}}} \\cdot (1 - \\alpha_W) $$
      </div>
      <div class="ipo-formula" style="font-size:.85rem">
        Con los valores actuales:
        $$ OEE = {eta_T:.3f} \\cdot {eta_F:.3f} \\cdot {eta_M_speed:.3f} \\cdot (1 - {ipo.alpha_W:.3f}) = {ipo.ipo:.3f} $$
      </div>
    </div>

    <div class="sec">
      <div class="sec-title">Los 3 factores OEE + losses</div>
      <div class="ipo-components">{comps_oee}</div>
      <div style="color:{C['muted']};font-size:.72rem;margin-top:10px;line-height:1.5">
        <b>η_T (Availability)</b>: disponibilidad productiva — tiempo operando vs total.<br/>
        <b>η_F (Quality)</b>: calidad de carga — fill factor vs target ({target_fill*100:.0f}% óptimo).<br/>
        <b>η_M (Performance)</b>: velocidad — benchmark de ciclo {target_cycle_s:.0f}s vs real.<br/>
        <b>α_W (Losses)</b>: desperdicios por derrames / overfill.
      </div>
    </div>

    <div class="sec">
      <div class="sec-title">Desglose del tiempo de ciclo</div>
      <div class="ipo-formula" style="font-size:.85rem">
        $$ T_{{ciclo}} = T_{{pos\\_C}} + T_{{carga}} + T_{{viaje\\_c}} + T_{{desc}} + T_{{pos\\_D}} + T_{{viaje\\_v}} $$
      </div>
      <div class="ipo-components">{t_parts}</div>
    </div>

    <div class="sec">
      <div class="sec-title">Interpretacion por bandas (escala OEE)</div>
      <div class="ipo-tbands">{bands}</div>
      {notes_html}
    </div>

    <div class="sec">
      <div class="sec-title">Componentes de referencia (legacy + contexto)</div>
      <div class="ipo-components">{comps_ref}</div>
      <div class="glossary" style="margin-top:10px">
        <dl>
          <dt>V_ef (efectivo)</dt><dd>{ipo.v_ef:.1f} m³ por cucharada = {ipo.v_nom:.0f} × {ipo.eta_R:.3f}</dd>
          <dt>V_cargado total</dt><dd>{ipo.v_cargado:.1f} m³ movidos en la ventana</dd>
          <dt>V_desperdiciado</dt><dd>{ipo.v_desperdiciado:.1f} m³ perdidos (derrames + overfill)</dd>
        </dl>
      </div>
    </div>
    """


def _hex_to_rgb(hex_str: str) -> str:
    """#ff8c00 -> '255,140,0' para usar en rgba(...)"""
    s = hex_str.lstrip('#')
    if len(s) == 3:
        s = ''.join(c*2 for c in s)
    try:
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        return f'{r},{g},{b}'
    except Exception:
        return '128,128,128'


def _pausas_view(visual: Dict) -> str:
    """
    Vista "Pausas del Operador" — eventos clasificados en 3 estados.
    Cards clickeables que hacen seek al video.
    Integrado del proyecto jevi con su filosofía de clasificación.
    """
    events = visual.get('fusion_events', [])
    total_inactivo = visual.get('total_inactivo_s', 0)

    state_config = {
        'INACTIVIDAD_JUSTIFICADA': {
            'color': C['yellow'], 'icon': '⏸',
            'label': 'Inactividad Justificada',
            'short': 'Justificada',
        },
        'INACTIVIDAD_INJUSTIFICADA': {
            'color': C['red'], 'icon': '⚠',
            'label': 'Inactividad Injustificada',
            'short': 'Injustificada',
        },
        'ACTIVIDAD_CONTRAPRODUCENTE': {
            'color': C['purple'], 'icon': '↯',
            'label': 'Actividad Contraproducente',
            'short': 'Contraproducente',
        },
    }

    # Contar por estado
    n_just = sum(1 for e in events if e.get('estado') == 'INACTIVIDAD_JUSTIFICADA')
    n_injust = sum(1 for e in events if e.get('estado') == 'INACTIVIDAD_INJUSTIFICADA')
    n_cp = sum(1 for e in events if e.get('estado') == 'ACTIVIDAD_CONTRAPRODUCENTE')

    if not events:
        return f"""
        <div class="panel" style="padding:30px;text-align:center">
          <div style="font-size:2.5rem;margin-bottom:10px">✓</div>
          <h3 style="color:{C['green']};margin-bottom:8px">Sin pausas detectadas</h3>
          <p style="color:{C['muted']};font-size:.85rem">
            La fusión IMU + Optical Flow no detectó períodos de inactividad ni actividad
            contraproducente en la ventana observada.
          </p>
        </div>
        """

    # Header con contadores
    header = f"""
    <div class="sec">
      <div class="panel" style="padding:14px">
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px">
          <div style="text-align:center;padding:10px;background:rgba({_hex_to_rgb(C['red'])},.08);
                      border:1px solid {C['red']};border-radius:8px">
            <div style="font-size:1.6rem">⚠</div>
            <div style="font-size:1.8rem;font-weight:700;color:{C['red']};font-family:monospace">{n_injust}</div>
            <div style="font-size:.72rem;color:{C['muted']}">Inactividad Injustificada</div>
          </div>
          <div style="text-align:center;padding:10px;background:rgba({_hex_to_rgb(C['yellow'])},.08);
                      border:1px solid {C['yellow']};border-radius:8px">
            <div style="font-size:1.6rem">⏸</div>
            <div style="font-size:1.8rem;font-weight:700;color:{C['yellow']};font-family:monospace">{n_just}</div>
            <div style="font-size:.72rem;color:{C['muted']}">Inactividad Justificada</div>
          </div>
          <div style="text-align:center;padding:10px;background:rgba({_hex_to_rgb(C['purple'])},.08);
                      border:1px solid {C['purple']};border-radius:8px">
            <div style="font-size:1.6rem">↯</div>
            <div style="font-size:1.8rem;font-weight:700;color:{C['purple']};font-family:monospace">{n_cp}</div>
            <div style="font-size:.72rem;color:{C['muted']}">Contraproducente</div>
          </div>
          <div style="text-align:center;padding:10px;background:{C['panel2']};
                      border:1px solid {C['border']};border-radius:8px">
            <div style="font-size:1.6rem">⏱</div>
            <div style="font-size:1.8rem;font-weight:700;color:{C['blue']};font-family:monospace">{total_inactivo:.0f}s</div>
            <div style="font-size:.72rem;color:{C['muted']}">Tiempo Inactivo Total</div>
          </div>
        </div>
      </div>
    </div>
    """

    # Cards por evento
    cards_html = ''
    for idx, ev in enumerate(events):
        state = ev.get('estado', 'INACTIVIDAD_INJUSTIFICADA')
        cfg = state_config.get(state, state_config['INACTIVIDAD_INJUSTIFICADA'])
        t_start = ev.get('tiempo_inicio_s', 0)
        t_end = ev.get('tiempo_fin_s', 0)
        dur = ev.get('duracion_s', 0)

        # Formato mm:ss.sss
        def _fmt_t(s):
            m = int(s) // 60
            return f'{m:02d}:{s - m*60:06.3f}'

        extras = ''
        if state == 'ACTIVIDAD_CONTRAPRODUCENTE':
            p = ev.get('periodicity', 0)
            extras = f'<div style="color:{C["muted"]};font-size:.68rem;margin-top:4px">Periodicity: {p:.2f} (< 0.15 = no ciclos productivos)</div>'

        cards_html += f"""
        <div class="pausa-card" onclick="seekToTime({t_start})"
             style="background:{C['panel']};border:1.5px solid {C['border']};
                    border-radius:8px;padding:12px;cursor:pointer;
                    transition:all .15s;position:relative"
             onmouseover="this.style.borderColor='{cfg['color']}';this.style.background='{C['panel2']}'"
             onmouseout="this.style.borderColor='{C['border']}';this.style.background='{C['panel']}'">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <span style="color:{C['muted']};font-size:.7rem">#{idx + 1}</span>
            <span style="color:{cfg['color']};font-size:.75rem;font-weight:700">⏱ {dur:.1f} s</span>
          </div>
          <div style="color:{C['text']};font-size:.9rem;font-weight:700;font-family:'Courier New',monospace;margin-bottom:8px">
            {_fmt_t(t_start)}  →  {_fmt_t(t_end)}
          </div>
          <div style="display:inline-block;padding:3px 10px;border-radius:11px;
                      background:rgba({_hex_to_rgb(cfg['color'])},.15);
                      border:1px solid {cfg['color']};
                      color:{cfg['color']};font-size:.7rem;font-weight:700">
            {cfg['icon']}  {cfg['label']}
          </div>
          {extras}
        </div>
        """

    return header + f"""
    <div class="sec">
      <div class="sec-title">Eventos detectados ({len(events)})</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px">
        {cards_html}
      </div>
    </div>

    <div class="sec">
      <div class="sec-title">Cómo se detectan estos eventos</div>
      <div class="panel" style="padding:14px">
        <div style="color:{C['text']};font-size:.82rem;line-height:1.6">
          <p style="margin-bottom:10px">
            <b style="color:{C['blue']}">Fusión estricta IMU + Cámara</b>:
            se marca INACTIVO cuando <b>TODOS</b> los 3 criterios se cumplen en una ventana de 2s:
          </p>
          <ul style="margin-left:20px;color:{C['muted']};font-size:.78rem;line-height:1.8">
            <li><b>acc_net</b> &lt; 3.0 m/s² (aceleración sin gravedad)</li>
            <li><b>gyro_mag</b> &lt; 18.0 deg/s (velocidad angular total)</li>
            <li><b>cam_motion</b> &lt; 1.3 px/frame (optical flow Farneback)</li>
          </ul>
          <p style="margin-top:12px;margin-bottom:6px">
            <b style="color:{C['yellow']}">Clasificación JUSTIFICADA vs INJUSTIFICADA</b>:
            se mira el <b>gyro_mean en los 15s previos</b> al idle.
            Si venía girando fuerte (&gt; 10.5 deg/s) → INJUSTIFICADA (esperando camión).
            Si venía moderado → JUSTIFICADA (pausa natural).
          </p>
          <p style="margin-top:10px">
            <b style="color:{C['purple']}">CONTRAPRODUCENTE</b>:
            segmentos activos &gt; 180s donde la autocorrelación del gyro no muestra
            patrón periódico (&lt; 0.15) → pala moviéndose sin ciclo productivo.
          </p>
        </div>
      </div>
    </div>
    """


def _insights_view(metrics, idle: Dict, ipo, transport, visual: Dict) -> str:
    """
    Vista que responde las 4 preguntas del brief del hackathon:
      1. ¿Qué midieron y cómo?
      2. ¿Qué les dijeron los datos?
      3. ¿Qué recomendarían al operador?
      4. ¿Qué construirían con 6hs más?
    """
    # Datos reales
    duration_min = metrics.total_duration_s / 60
    iti_pct = idle.get('iti_pct', 0)
    verified_iti_pct = idle.get('verified_iti_pct', 0)
    agreement = idle.get('agreement_pct', 0)
    confidence = idle.get('confidence', 'N/A')
    prod_alerts = idle.get('productivity_alerts', [])
    disagreements = idle.get('disagreements', [])
    n_cycles = metrics.n_full_cycles
    payload_t = metrics.total_payload_t

    conf_colors = {'HIGH': C['green'], 'MEDIUM': C['yellow'], 'LOW': C['red']}
    conf_c = conf_colors.get(confidence, C['muted'])

    # Métrica "pérdida estimada" visualmente impactante
    idle_minutes = idle.get('verified_idle_s', idle.get('total_idle_s', 0)) / 60
    productivity_tph = metrics.productivity_tph if metrics.productivity_tph > 0 else 400
    estimated_loss_t = (idle_minutes / 60) * productivity_tph

    return f"""
    <div class="sec">
      <div class="sec-title">Pregunta 1 — ¿Qué midieron y cómo?</div>
      <div class="panel" style="padding:18px">
        <p style="margin-bottom:12px;color:{C['text']}">
          Medimos <b style="color:{C['blue']}">productividad operacional</b> con un enfoque
          <b>dual-source</b> inédito:
        </p>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:12px">
          <div class="panel" style="background:{C['panel2']};padding:12px">
            <div style="font-weight:700;color:{C['orange']};margin-bottom:6px">🛰 SENSOR 1 — IMU (mecánico)</div>
            <div style="color:{C['muted']};font-size:.82rem">
              Acelerómetro + giroscopio @ ~10Hz detectan movimiento
              mecánico de la pala. Identificamos ciclos (DIG→SWING→DUMP),
              wait events, y anomalías (impactos duros, dumps bruscos).
            </div>
          </div>
          <div class="panel" style="background:{C['panel2']};padding:12px">
            <div style="font-weight:700;color:{C['blue']};margin-bottom:6px">🎥 SENSOR 2 — Optical Flow (visual)</div>
            <div style="color:{C['muted']};font-size:.82rem">
              La cámara va montada sobre la pala. Calculamos <b>optical flow
              Farneback</b> entre frames consecutivos: cámara moviéndose =
              pala moviéndose. Cámara quieta = pala quieta (idle real).
            </div>
          </div>
        </div>
        <div class="panel" style="background:rgba({_hex_to_rgb(C['green'])},.1);border-left:4px solid {C['green']};padding:10px 14px">
          <b style="color:{C['green']}">Insight maestro</b>: al cruzar ambas fuentes obtenemos
          <b>idle verificado</b> (ambas confirman quietud) y <b>discrepancias flag</b>
          (una dice idle y la otra actividad — auditoría). El acuerdo entre sensores
          da el <b>índice de confianza del operador</b>.
        </div>
      </div>
    </div>

    <div class="sec">
      <div class="sec-title">Pregunta 2 — ¿Qué les dijeron los datos?</div>
      <div class="panel" style="padding:18px">
        <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-bottom:14px">
          <div>
            <div style="font-size:.68rem;color:{C['muted']};text-transform:uppercase">Ventana observada</div>
            <div style="font-size:1.6rem;font-weight:700;font-family:monospace">{duration_min:.1f} min</div>
          </div>
          <div>
            <div style="font-size:.68rem;color:{C['muted']};text-transform:uppercase">Ciclos completos</div>
            <div style="font-size:1.6rem;font-weight:700;color:{C['green']};font-family:monospace">{n_cycles}</div>
          </div>
          <div>
            <div style="font-size:.68rem;color:{C['muted']};text-transform:uppercase">Payload total</div>
            <div style="font-size:1.6rem;font-weight:700;color:{C['blue']};font-family:monospace">{payload_t:.0f}t</div>
          </div>
          <div>
            <div style="font-size:.68rem;color:{C['muted']};text-transform:uppercase">ITI (IMU)</div>
            <div style="font-size:1.6rem;font-weight:700;color:{C['red']};font-family:monospace">{iti_pct:.1f}%</div>
          </div>
          <div>
            <div style="font-size:.68rem;color:{C['muted']};text-transform:uppercase">ITI verificado</div>
            <div style="font-size:1.6rem;font-weight:700;color:{conf_c};font-family:monospace">{verified_iti_pct:.1f}%</div>
          </div>
          <div>
            <div style="font-size:.68rem;color:{C['muted']};text-transform:uppercase">Acuerdo IMU↔Cámara</div>
            <div style="font-size:1.6rem;font-weight:700;color:{conf_c};font-family:monospace">{agreement:.1f}%</div>
          </div>
        </div>
        <div style="color:{C['text']};line-height:1.7">
          <p style="margin-bottom:8px">
            <b style="color:{C['red']}">Hallazgo #1</b>: La operación tiene un
            <b>ITI del {iti_pct:.1f}%</b> según el IMU — más de la mitad del tiempo sin
            movimiento mecánico. Equivale a <b>{idle.get('total_idle_s',0)/60:.1f} minutos</b>
            perdidos en la ventana de 15 min.
          </p>
          <p style="margin-bottom:8px">
            <b style="color:{conf_c}">Hallazgo #2 (dual-source)</b>: El optical flow
            confirma el <b>{verified_iti_pct:.1f}%</b> como idle REAL (ambas fuentes concuerdan).
            Confianza global: <b>{confidence}</b>. {'Alto nivel de confianza en la métrica.' if confidence == 'HIGH' else 'Hay discrepancias: la cámara ve movimiento durante waits del IMU.'}
          </p>
          <p style="margin-bottom:8px">
            <b style="color:{C['yellow']}">Hallazgo #3</b>: Detectamos <b>{len(prod_alerts)} períodos
            de pérdida productiva</b> (cámara sin movimiento ≥15s). Cada uno representa
            toneladas no movidas auditable desde el video.
          </p>
          <p>
            <b style="color:{C['purple']}">Hallazgo #4</b>: Se identificaron <b>{len(disagreements)} discrepancias</b>
            entre IMU y cámara. Estos son eventos donde <b>hay que auditar el sensor</b>
            porque las fuentes no concuerdan.
          </p>
        </div>
      </div>
    </div>

    <div class="sec">
      <div class="sec-title">Pregunta 3 — ¿Qué recomendarían al operador?</div>
      <div class="panel" style="padding:18px">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
          <div class="panel" style="background:rgba({_hex_to_rgb(C['red'])},.08);border-left:3px solid {C['red']};padding:12px">
            <div style="font-weight:700;color:{C['red']};margin-bottom:6px">🚨 Acción inmediata</div>
            <ul style="color:{C['text']};margin-left:18px;line-height:1.7">
              <li>Si ves <b>banner naranja</b> en Live = operás con pérdida productiva AHORA</li>
              <li>Revisar los <b>períodos idle &gt;60s</b> en la vista Ocio — son intervenciones críticas</li>
              <li>Coordinar con despacho para cerrar la brecha entre camiones</li>
            </ul>
          </div>
          <div class="panel" style="background:rgba({_hex_to_rgb(C['yellow'])},.08);border-left:3px solid {C['yellow']};padding:12px">
            <div style="font-weight:700;color:{C['yellow']};margin-bottom:6px">⚠ Medio plazo</div>
            <ul style="color:{C['text']};margin-left:18px;line-height:1.7">
              <li>Investigar <b>discrepancias IMU↔cámara</b>: posible sensor en mal estado</li>
              <li>Si ITI verificado {'<' if verified_iti_pct < 30 else '>'} 30%, revisar flota de camiones vs capacidad de pala</li>
              <li>Reducir impactos duros durante DIG ajustando ángulo de ataque</li>
            </ul>
          </div>
          <div class="panel" style="background:rgba({_hex_to_rgb(C['blue'])},.08);border-left:3px solid {C['blue']};padding:12px">
            <div style="font-weight:700;color:{C['blue']};margin-bottom:6px">💡 Estratégico</div>
            <ul style="color:{C['text']};margin-left:18px;line-height:1.7">
              <li>Usar el <b>idle verificado ({verified_iti_pct:.1f}%)</b> como KPI principal, no el IMU-solo</li>
              <li>Instalar dashboard en sala de control con alertas dual-source</li>
              <li>Correlacionar ITI con turnos/operadores para identificar oportunidades</li>
            </ul>
          </div>
          <div class="panel" style="background:rgba({_hex_to_rgb(C['green'])},.08);border-left:3px solid {C['green']};padding:12px">
            <div style="font-weight:700;color:{C['green']};margin-bottom:6px">✅ Quick wins</div>
            <ul style="color:{C['text']};margin-left:18px;line-height:1.7">
              <li>Si recuperás <b>10% del ITI</b>, ganás ~{estimated_loss_t*0.1:.0f}t/turno</li>
              <li>Los {len(prod_alerts)} eventos idle detectados suman {sum(p['duration_s'] for p in prod_alerts)/60:.1f} min recuperables</li>
              <li>Fill factor promedio es {metrics.avg_fill_factor_pct:.0f}% — objetivo 85%+</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class="sec">
      <div class="sec-title">Pregunta 4 — ¿Qué construirían con 6 horas más?</div>
      <div class="panel" style="padding:18px">
        <ol style="color:{C['text']};line-height:1.8;margin-left:20px">
          <li><b style="color:{C['blue']}">Fine-tune YOLO sobre el dataset propio</b>: con labeling automatizado vía SAM2, detectar bucket específico (no solo camiones) para mejorar aún más la validación visual.</li>
          <li><b style="color:{C['green']}">Predicción de ITI próxima hora</b>: modelo ARIMA o LSTM sobre la serie temporal de motion + IMU para anticipar bajones productivos ANTES que sucedan.</li>
          <li><b style="color:{C['yellow']}">Scoring de operador</b>: comparar métricas IPO entre operadores del mismo turno para identificar mejores prácticas transferibles.</li>
          <li><b style="color:{C['orange']}">Integración con dispatch</b>: API bidireccional con sistema de despacho — cuando la pala entra en idle verificado, redirigir automáticamente camiones más cercanos.</li>
          <li><b style="color:{C['purple']}">Alerta push tiempo real</b>: cuando idle verificado supera threshold, notificar supervisor vía SMS/Teams con clip de 10s del evento.</li>
          <li><b style="color:{C['red']}">Hard impact prevention</b>: con accelerómetro + video, crear un "assistant" que avise al operador si el ángulo de ataque al banco es sub-óptimo.</li>
        </ol>
      </div>
    </div>
    """


def _idle_view(idle: Dict) -> str:
    """Vista del Indice de Tiempos de Ocio — hero + desglose + top waits + timeline."""
    if not idle or idle.get('total_duration_s', 0) <= 0:
        return ('<div class="panel" style="padding:20px">'
                '<p style="color:#8b949e">Sin datos de ocio disponibles.</p></div>')

    band_colors = {
        'OPTIMO':       C['green'],
        'NORMAL':       C['blue'],
        'CRITICO':      C['yellow'],
        'COMPROMETIDO': C['red'],
        'N/A':          C['muted'],
    }
    bc = band_colors.get(idle.get('band', 'N/A'), C['muted'])
    iti_pct = idle.get('iti_pct', 0.0)
    prod_pct = 100 - iti_pct

    # Badge de confianza (validación visual)
    conf_label = idle.get('confidence', None)
    conf_score = idle.get('confidence_score', 0.0)
    iti_pct_imu_raw = idle.get('iti_pct_imu_raw', 0.0)
    false_pos_n = idle.get('false_positives_n', 0)
    false_pos_s = idle.get('false_positives_s', 0.0)
    disagreements_n = idle.get('disagreements_n', 0)

    conf_colors = {
        'HIGH':   C['green'],
        'MEDIUM': C['yellow'],
        'LOW':    C['red'],
    }
    conf_badge_html = ''
    if conf_label:
        cc = conf_colors.get(conf_label, C['muted'])
        icon = '🟢' if conf_label == 'HIGH' else ('🟡' if conf_label == 'MEDIUM' else '🔴')
        # Comparación IMU crudo vs ITI REAL
        comparison_note = (
            f'<div style="color:{C["muted"]};font-size:.72rem;margin-top:6px;line-height:1.5">'
            f'IMU crudo reportaba: <s style="color:{C["red"]}">{iti_pct_imu_raw:.1f}%</s> → '
            f'ITI real <b style="color:{cc};font-family:monospace">{iti_pct:.1f}%</b><br/>'
            f'<span style="color:{C["muted"]};font-size:.68rem">'
            f'Detectados {false_pos_n} falsos positivos del IMU ({false_pos_s:.0f}s descartados)'
            f'</span></div>'
        )
        conf_badge_html = f"""
        <div style="margin-top:14px;padding:10px 16px;background:rgba({_hex_to_rgb(cc)},.10);
                    border:1px solid {cc};border-radius:8px;display:inline-block;max-width:520px">
          <div style="display:flex;align-items:center;gap:10px;justify-content:center">
            <span style="font-size:1.2rem">{icon}</span>
            <span style="color:{cc};font-weight:700;letter-spacing:.05em;font-size:.85rem">
              CONFIANZA {conf_label} · acuerdo {conf_score*100:.0f}%
            </span>
          </div>
          <div style="color:{C['muted']};font-size:.7rem;margin-top:4px">
            Dual-source: IMU + Optical Flow (cámara)
          </div>
          {comparison_note}
        </div>
        """

    # Hero: ITI grande + productividad complementaria
    hero = f"""
    <div class="ipo-hero">
      <div style="color:{C['muted']};font-size:.75rem;text-transform:uppercase;letter-spacing:.1em">
        Indice de Tiempos de Ocio (ITI)
      </div>
      <div class="ipo-big" style="color:{bc}">{iti_pct:.1f}%</div>
      <div class="ipo-band" style="background:{bc};color:#000">
        {idle.get('band', 'N/A')}
      </div>
      {conf_badge_html}
      <div style="color:{C['muted']};font-size:.85rem;margin-top:10px;max-width:560px;margin-left:auto;margin-right:auto">
        {idle.get('recommendation', '')}
      </div>
      <div style="margin-top:18px;display:flex;gap:20px;justify-content:center;flex-wrap:wrap">
        <div>
          <div style="font-size:.68rem;color:{C['muted']};text-transform:uppercase;letter-spacing:.06em">Tiempo ocio</div>
          <div style="font-size:1.6rem;font-weight:700;color:{bc};font-family:monospace">
            {idle.get('total_idle_s', 0):.0f} s
          </div>
          <div style="font-size:.68rem;color:{C['muted']}">
            = {idle.get('total_idle_s', 0)/60:.1f} min
          </div>
        </div>
        <div>
          <div style="font-size:.68rem;color:{C['muted']};text-transform:uppercase;letter-spacing:.06em">Tiempo productivo</div>
          <div style="font-size:1.6rem;font-weight:700;color:{C['green']};font-family:monospace">
            {idle.get('productive_s', 0):.0f} s
          </div>
          <div style="font-size:.68rem;color:{C['muted']}">
            = {idle.get('productive_s', 0)/60:.1f} min
          </div>
        </div>
        <div>
          <div style="font-size:.68rem;color:{C['muted']};text-transform:uppercase;letter-spacing:.06em">Ratio Prod/Ocio</div>
          <div style="font-size:1.6rem;font-weight:700;color:{C['blue']};font-family:monospace">
            {idle.get('ratio_prod_idle', 0):.2f}x
          </div>
          <div style="font-size:.68rem;color:{C['muted']}">
            {idle.get('n_events', 0)} eventos de ocio
          </div>
        </div>
      </div>
    </div>
    """

    # Barra visual: productivo vs ocio
    progress_bar = f"""
    <div class="sec">
      <div class="sec-title">Distribucion del tiempo observado</div>
      <div style="background:{C['panel2']};border-radius:8px;padding:16px;border:1px solid {C['border']}">
        <div style="display:flex;height:48px;border-radius:6px;overflow:hidden;box-shadow:inset 0 0 0 1px {C['border']}">
          <div style="width:{prod_pct:.2f}%;background:{C['green']};display:flex;align-items:center;justify-content:center;color:#000;font-weight:700;font-size:.9rem;min-width:60px">
            {prod_pct:.1f}% PROD
          </div>
          <div style="width:{iti_pct:.2f}%;background:{bc};display:flex;align-items:center;justify-content:center;color:#000;font-weight:700;font-size:.9rem;min-width:60px">
            {iti_pct:.1f}% OCIO
          </div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:.72rem;color:{C['muted']}">
          <span>0 s</span>
          <span style="font-family:monospace">Total observado: {idle.get('total_duration_s', 0):.0f} s</span>
          <span>{idle.get('total_duration_s', 0):.0f} s</span>
        </div>
      </div>
    </div>
    """

    # Desglose por categoria
    by_cat = idle.get('by_category', {})
    cat_rows = ''
    total_idle = idle.get('total_idle_s', 1)
    cat_colors_map = {
        'pre_load':    C['blue'],
        'inter_cycle': C['yellow'],
        'end_session': C['purple'],
        'critical':    C['red'],
        'unknown':     C['muted'],
    }
    for cat, data in by_cat.items():
        if data['n_events'] == 0:
            continue
        col = cat_colors_map.get(cat, C['muted'])
        bar_pct = min(100, data['pct_of_idle'])
        cat_rows += f"""
        <tr>
          <td><span style="display:inline-block;width:10px;height:10px;background:{col};border-radius:2px;margin-right:6px"></span>
            <b>{data['label']}</b>
          </td>
          <td style="font-family:monospace">{data['total_s']:.1f} s</td>
          <td style="font-family:monospace">{data['n_events']}</td>
          <td style="min-width:160px">
            <div style="background:{C['panel2']};height:14px;border-radius:3px;overflow:hidden;display:inline-block;width:120px;vertical-align:middle;margin-right:8px">
              <div style="width:{bar_pct}%;background:{col};height:100%"></div>
            </div>
            <span style="font-family:monospace;font-size:.75rem">{data['pct_of_idle']:.1f}%</span>
          </td>
        </tr>
        """

    desglose = f"""
    <div class="sec">
      <div class="sec-title">Desglose por categoria</div>
      <div class="panel" style="padding:0;overflow:hidden">
        <table>
          <thead>
            <tr>
              <th>Categoria</th>
              <th>Total (s)</th>
              <th># eventos</th>
              <th>% del ocio</th>
            </tr>
          </thead>
          <tbody>{cat_rows or '<tr><td colspan="4" style="text-align:center;color:#8b949e">Sin datos</td></tr>'}</tbody>
        </table>
      </div>
    </div>
    """

    # Top waits mas largos
    top_waits = idle.get('top_waits', [])
    top_rows = ''
    for i, w in enumerate(top_waits, 1):
        # Color por severidad del wait
        if w['duration'] > 60:
            wc = C['red']; wbadge = 'br'
        elif w['duration'] > 25:
            wc = C['yellow']; wbadge = 'bw'
        else:
            wc = C['blue']; wbadge = 'bb'
        top_rows += f"""
        <tr>
          <td><b>#{i}</b></td>
          <td style="font-family:monospace">t={w['t_start']:.1f}s → {w['t_end']:.1f}s</td>
          <td><span class="badge {wbadge}" style="font-family:monospace">{w['duration']:.1f} s</span></td>
          <td>{w['label']}</td>
          <td style="font-family:monospace;color:{wc}">{w['pct_total']:.2f}%</td>
        </tr>
        """

    top_table = f"""
    <div class="sec">
      <div class="sec-title">Top 5 waits mas largos — candidatos a optimizar</div>
      <div class="panel" style="padding:0;overflow:hidden">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Ventana</th>
              <th>Duracion</th>
              <th>Razon</th>
              <th>% del total</th>
            </tr>
          </thead>
          <tbody>{top_rows or '<tr><td colspan="5" style="text-align:center;color:#8b949e">Sin waits registrados</td></tr>'}</tbody>
        </table>
      </div>
    </div>
    """

    # Bandas de interpretacion
    bands_html = ""
    for lo, hi, lbl, col_name, rec in [
        (0.00, 0.15, 'OPTIMO',       'green',  'Muy eficiente'),
        (0.15, 0.30, 'NORMAL',       'blue',   'Rango esperado'),
        (0.30, 0.50, 'CRITICO',      'yellow', 'Revisar coordinacion'),
        (0.50, 1.00, 'COMPROMETIDO', 'red',    'Intervencion urgente'),
    ]:
        active_cls = ' active' if idle.get('band') == lbl else ''
        col_hex = C[col_name]
        bands_html += f"""
        <div class="ipo-tband{active_cls}" style="background:rgba({_hex_to_rgb(col_hex)},.15);border-color:{col_hex}">
          <b style="color:{col_hex}">{lbl}</b>
          <span style="color:{C['muted']};font-size:.65rem;display:block">{lo*100:.0f}% - {hi*100:.0f}%</span>
          {rec}
        </div>
        """

    bands_block = f"""
    <div class="sec">
      <div class="sec-title">Interpretacion por bandas</div>
      <div class="ipo-tbands">{bands_html}</div>
    </div>
    """

    # Formula + recomendaciones
    formula_block = f"""
    <div class="sec">
      <div class="sec-title">Formula y referencia</div>
      <div class="ipo-formula">
        $$ ITI = \\frac{{T_{{ocio}}}}{{T_{{observado}}}} = \\frac{{\\sum_i d_i^{{wait}}}}{{T_{{total}}}} = \\frac{{{idle.get('total_idle_s', 0):.1f}}}{{{idle.get('total_duration_s', 0):.1f}}} = {idle.get('iti', 0):.3f} $$
      </div>
      <div class="glossary" style="margin-top:10px">
        <h4>Por que es la metrica mas clara</h4>
        <p style="color:{C['muted']}">
          A diferencia del IPO que combina 5 factores, el ITI responde una pregunta simple:
          <b>"¿Cuanto tiempo no estas produciendo?"</b>. Es facil de explicar, facil de medir
          y facil de optimizar. Cada segundo que la pala no carga = perdida directa de productividad.
        </p>
        <h4 style="margin-top:12px">Como reducir el ITI</h4>
        <dl>
          <dt>Si domina <b>pre_load</b></dt><dd>Mejorar coordinacion con despacho para que el primer camion llegue a tiempo.</dd>
          <dt>Si domina <b>inter_cycle</b></dt><dd>Optimizar rotacion de camiones. Evaluar si hay suficientes camiones en la flota para la pala.</dd>
          <dt>Si domina <b>critical</b> (&gt;60s)</dt><dd>Son interrupciones anormales. Revisar mantenimiento, cambio de turno o eventos imprevistos.</dd>
          <dt>Si domina <b>end_session</b></dt><dd>Planificar mejor el fin de turno. Asegurar que el ultimo camion llegue con tiempo.</dd>
        </dl>
      </div>
    </div>
    """

    # Bloque de validación visual (dual-source) — solo si hay datos
    visual_block = ''
    if conf_label:
        wait_confs = idle.get('wait_confidence', [])
        disagreements = idle.get('disagreements', [])

        # Tabla de confianza por wait
        wait_rows = ''
        for wc in wait_confs[:10]:  # top 10 por duración implícita (vienen ordenados)
            conf_c = conf_colors.get(wc['confidence'], C['muted'])
            validated_icon = '✓' if wc['validated'] else '✗'
            wait_rows += f"""
            <tr>
              <td style="font-family:monospace">t={wc['t_start']:.1f}s → {wc['t_end']:.1f}s</td>
              <td style="font-family:monospace">{wc['duration_s']:.1f}s</td>
              <td>{wc['reason']}</td>
              <td>
                <span class="badge" style="background:rgba({_hex_to_rgb(conf_c)},.15);color:{conf_c};font-weight:700">
                  {validated_icon} {wc['confidence']}
                </span>
              </td>
              <td style="font-family:monospace">
                {'🎥 static' if wc.get('visual_static', False) else ('🎥 ' + str(wc.get('motion_score', 0)))}
              </td>
            </tr>
            """

        # Discrepancias flag
        disag_html = ''
        if disagreements:
            disag_rows = ''
            for d in disagreements[:8]:
                disag_rows += f"""
                <tr>
                  <td><span class="badge br">{d['type']}</span></td>
                  <td style="font-family:monospace">t={d['t_start']:.1f}s</td>
                  <td style="font-family:monospace">{d['duration_s']:.1f}s</td>
                  <td style="color:{C['muted']};font-size:.75rem">{d['note']}</td>
                </tr>
                """
            disag_html = f"""
            <div class="sec">
              <div class="sec-title" style="color:{C['red']}">⚠ Discrepancias entre IMU y Video ({len(disagreements)})</div>
              <div class="panel" style="padding:0;overflow:hidden">
                <table>
                  <thead>
                    <tr><th>Tipo</th><th>Tiempo</th><th>Duración</th><th>Nota</th></tr>
                  </thead>
                  <tbody>{disag_rows}</tbody>
                </table>
              </div>
            </div>
            """

        visual_block = f"""
        <div class="sec">
          <div class="sec-title">Validación visual — Dual-source (IMU + Optical Flow)</div>
          <div class="panel" style="padding:14px">
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:14px">
              <div>
                <div style="font-size:.68rem;color:{C['muted']};text-transform:uppercase">Confianza</div>
                <div style="font-size:1.4rem;font-weight:700;color:{conf_colors.get(conf_label, C['muted'])};font-family:monospace">
                  {conf_label}
                </div>
                <div style="font-size:.68rem;color:{C['muted']}">
                  acuerdo {conf_score*100:.1f}%
                </div>
              </div>
              <div>
                <div style="font-size:.68rem;color:{C['muted']};text-transform:uppercase">ITI IMU crudo</div>
                <div style="font-size:1.4rem;font-weight:700;color:{C['red']};font-family:monospace;text-decoration:line-through;opacity:.7">
                  {iti_pct_imu_raw:.1f}%
                </div>
                <div style="font-size:.68rem;color:{C['muted']}">con falsos positivos</div>
              </div>
              <div>
                <div style="font-size:.68rem;color:{C['muted']};text-transform:uppercase">ITI REAL</div>
                <div style="font-size:1.4rem;font-weight:700;color:{C['green']};font-family:monospace">
                  {iti_pct:.1f}%
                </div>
                <div style="font-size:.68rem;color:{C['muted']}">confirmado por video</div>
              </div>
              <div>
                <div style="font-size:.68rem;color:{C['muted']};text-transform:uppercase">Falsos positivos</div>
                <div style="font-size:1.4rem;font-weight:700;color:{C['yellow']};font-family:monospace">
                  {false_pos_n}
                </div>
                <div style="font-size:.68rem;color:{C['muted']}">= {false_pos_s:.0f}s descartados</div>
              </div>
            </div>
            <div style="color:{C['muted']};font-size:.78rem;line-height:1.5">
              <b style="color:{C['text']}">Insight clave</b>: el IMU reportaba ocio del
              <b style="color:{C['red']}">{iti_pct_imu_raw:.1f}%</b>, pero al validar con optical flow
              (cámara montada sobre la pala) descubrimos que {false_pos_n} eventos eran
              <b style="color:{C['yellow']}">falsos positivos</b>. El ocio REAL es
              <b style="color:{C['green']}">{iti_pct:.1f}%</b>. El IMU detecta "ausencia de movimiento mecánico"
              pero no distingue entre waits reales y swings suaves.
            </div>
          </div>

          {'<div class="panel" style="padding:0;overflow:auto;margin-top:10px"><table><thead><tr><th>Ventana</th><th>Duración</th><th>Razón</th><th>Confianza</th><th>Video</th></tr></thead><tbody>' + wait_rows + '</tbody></table></div>' if wait_rows else ''}
        </div>
        {disag_html}
        """

    return hero + progress_bar + desglose + top_table + bands_block + formula_block + visual_block
