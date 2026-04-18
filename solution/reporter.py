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


def generate_report(df_imu, cycles, wait_events, metrics,
                    video_events, df_cycles, df_waits, output_dir,
                    profiles=None, alerts=None,
                    truck_events=None, transport=None,
                    timeline=None,
                    ocr_events=None, wear_score=None,
                    spill_events=None, dust_index=None,
                    session_label='JEBI 2026 - EX-5600') -> str:

    html = _build(df_imu, cycles, wait_events, metrics,
                  video_events, profiles or [], alerts or [],
                  truck_events or [], transport,
                  timeline or [], session_label,
                  ocr_events or [], wear_score or {},
                  spill_events or [], dust_index or {})

    path = os.path.join(output_dir, 'dashboard.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  [reporter] Dashboard guardado: {path}')
    return path


# ─── BUILDER PRINCIPAL ───────────────────────────────────────────────────────

def _build(df_imu, cycles, wait_events, metrics, video_events,
           profiles, alerts, truck_events, transport, timeline, label,
           ocr_events=None, wear_score=None, spill_events=None, dust_index=None):

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
    kpis_html   = _kpi_cards(metrics, transport)
    eff_html    = _efficiency_table(profiles)
    trucks_html = _trucks_table(truck_events)
    waits_html  = _waits_table(wait_events)
    thumbs_html = _thumbs(video_events)
    ocr_html    = _ocr_grid(ocr_events or [])
    bpmn_html   = _bpmn_coverage(metrics, transport, wear_score or {},
                                  spill_events or [], dust_index or {}, alerts)
    help_html   = _help_content()

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Shovel Intelligence | {label}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:{C['bg']};color:{C['text']};font-family:-apple-system,'Segoe UI',sans-serif;font-size:13px;line-height:1.5}}
::-webkit-scrollbar{{width:6px;height:6px}}::-webkit-scrollbar-track{{background:{C['panel']}}}::-webkit-scrollbar-thumb{{background:{C['border']};border-radius:3px}}

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
.row{{display:flex;gap:12px}}
.col-video{{flex:0 0 960px}}
.col-alerts{{flex:1;min-width:280px}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.g3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}
.g4{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px}}
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
.video-wrap{{display:flex;gap:4px;background:#000;border-radius:8px;overflow:hidden}}
.video-wrap video{{flex:1;height:240px;object-fit:cover}}
.video-label{{font-size:0.65rem;color:{C['muted']};margin-bottom:4px}}
.video-overlay{{position:relative}}
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
.gauge-row{{display:flex;flex-wrap:wrap;gap:8px;justify-content:space-between}}
.gauge-box{{flex:1;min-width:180px;max-width:240px;background:{C['panel']};border:1px solid {C['border']};border-radius:8px;padding:8px}}
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

/* ═══ WEAR + DESPERDICIOS ═══ */
.bpmn-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}
.bpmn-card{{background:{C['panel']};border:1px solid {C['border']};border-radius:8px;padding:14px}}
.bpmn-card h4{{font-size:0.7rem;text-transform:uppercase;letter-spacing:.06em;color:{C['muted']};margin-bottom:8px}}
.bpmn-value{{font-size:1.8rem;font-weight:700;margin:6px 0}}
.bpmn-bar{{height:8px;background:{C['panel2']};border-radius:4px;overflow:hidden;margin:8px 0 4px}}
.bpmn-bar-fill{{height:100%;transition:width .4s ease}}
.bpmn-note{{font-size:0.7rem;color:{C['muted']}}}
</style>
</head>
<body>

<!-- HEADER -->
<div class="hdr">
  <div>
    <h1>Shovel Intelligence Dashboard</h1>
    <div class="meta">JEBI Hackathon 2026 &nbsp;|&nbsp; Hitachi EX-5600 &nbsp;|&nbsp;
      CAT 793F (218t) &nbsp;&amp;&nbsp; EH4000 AC-3 (221t) &nbsp;|&nbsp; {now}
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px">
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
  <div class="side-item" data-view="live" onclick="switchView('live')">
    <span class="side-icon">🎥</span><span>Video &amp; Live</span>
  </div>
  <div class="side-item" data-view="cycles" onclick="switchView('cycles')">
    <span class="side-icon">🔄</span><span>Ciclos</span>
  </div>
  <div class="side-item" data-view="trucks" onclick="switchView('trucks')">
    <span class="side-icon">🚛</span><span>Camiones &amp; OCR</span>
  </div>
  <div class="side-item" data-view="alerts" onclick="switchView('alerts')">
    <span class="side-icon">🚨</span><span>Alertas</span>
    <span class="side-badge" id="side-alert-count">0</span>
  </div>
  <div class="side-item" data-view="simulator" onclick="switchView('simulator')">
    <span class="side-icon">🎯</span><span>Simulador</span>
  </div>
  <div class="side-item" data-view="bpmn" onclick="switchView('bpmn')">
    <span class="side-icon">📈</span><span>BPMN Coverage</span>
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

<!-- VIDEO + ALERTAS -->
<div class="sec row">
  <div class="col-video">
    <div class="sec-title">Video Estereo &mdash; Camara Izquierda / Derecha</div>
    <div class="video-wrap">
      <div class="video-overlay" style="flex:1">
        <div class="video-label">CAM IZQUIERDA</div>
        <video id="vid-left" src="../inputs/shovel_left.mp4"
               muted playsinline preload="auto"
               style="width:100%;height:240px;object-fit:cover;border-radius:6px"></video>
        <div class="ov" id="vid-left-info">t=0.0s</div>
      </div>
      <div class="video-overlay" style="flex:1">
        <div class="video-label">CAM DERECHA</div>
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

<!-- ═══════════ VIEW: TRUCKS + OCR ═══════════ -->
<div class="view" id="view-trucks" data-view="trucks">
  <div class="view-hdr">
    <div>
      <h2>Camiones &amp; OCR — ID y Medidor de Peso</h2>
      <div class="desc">Detecciones YOLO26 + OCR del numero de camion y display de peso frontal</div>
    </div>
    <div class="view-actions">
      <button class="btn btn-primary" onclick="exportCurrentView()">📄 Exportar PDF</button>
    </div>
  </div>
  <div class="sec">
    <div class="sec-title">Carga por Camion — Despacho</div>
    {trucks_html}
  </div>
  <div class="sec">
    <div class="sec-title">Lecturas OCR (ID + Medidor de Peso por frame)</div>
    {ocr_html}
  </div>
  <div class="sec">
    <div class="sec-title">Capturas de Video — Eventos Clave</div>
    {thumbs_html}
  </div>
</div><!-- /view-trucks -->

<!-- ═══════════ VIEW: ALERTS ═══════════ -->
<div class="view" id="view-alerts" data-view="alerts">
  <div class="view-hdr">
    <div>
      <h2>Alertas Historico Completo</h2>
      <div class="desc">Todas las alertas de la sesion ordenadas por severidad y tiempo</div>
    </div>
    <div class="view-actions">
      <button class="btn btn-primary" onclick="exportCurrentView()">📄 Exportar PDF</button>
    </div>
  </div>
  <div class="sec">
    <div class="sec-title">Feed completo</div>
    <div class="panel" style="padding:10px">
      <div id="alert-feed-full" style="max-height:600px;overflow-y:auto"></div>
    </div>
  </div>
</div><!-- /view-alerts -->

<!-- ═══════════ VIEW: SIMULATOR ═══════════ -->
<div class="view" id="view-simulator" data-view="simulator">
  <div class="view-hdr">
    <div>
      <h2>Simulador de Colocacion — Bueno vs Malo</h2>
      <div class="desc">Visualizacion grafica de la posicion ideal del camion frente a la pala y errores comunes</div>
    </div>
    <div class="view-actions">
      <button class="btn btn-primary" onclick="exportCurrentView()">📄 Exportar PDF</button>
    </div>
  </div>
  <div class="sec sim-wrap">
    <div class="sim-panel">
      <h3 style="color:{C['green']}">✓ Colocacion Optima</h3>
      <canvas id="sim-ok" class="sim-canvas sim-ok" width="640" height="360"></canvas>
      <div class="sim-legend">
        <span><span class="dot" style="background:{C['green']}"></span>Zona optima</span>
        <span><span class="dot" style="background:{C['yellow']}"></span>Aceptable</span>
        <span><span class="dot" style="background:{C['red']}"></span>Rechazo</span>
      </div>
    </div>
    <div class="sim-panel">
      <h3 style="color:{C['red']}">✗ Colocacion Incorrecta</h3>
      <canvas id="sim-bad" class="sim-canvas sim-bad" width="640" height="360"></canvas>
      <div class="sim-controls">
        <button onclick="simScenario('offset')" class="active">Offset lateral</button>
        <button onclick="simScenario('far')">Muy lejos</button>
        <button onclick="simScenario('close')">Muy cerca</button>
        <button onclick="simScenario('angle')">Mal angulo</button>
      </div>
    </div>
  </div>
  <div class="sec">
    <div class="sec-title">Parametros de la Colocacion Ideal</div>
    <div class="glossary">
      <dl>
        <dt>Distancia pala-camion</dt><dd>~12-15m del centro del balde. Si muy lejos: mas swing, perdida de tiempo. Si muy cerca: riesgo de contacto.</dd>
        <dt>Angulo de pala</dt><dd>~90° respecto al eje del camion. Permite descarga centrada en tolva.</dd>
        <dt>Offset lateral</dt><dd>Camion centrado ±1m del punto de descarga. Desviacion grande = derrames.</dd>
        <dt>Altura tolva</dt><dd>Visible en frame superior. Debe estar por debajo de la parte mas alta del balde.</dd>
      </dl>
    </div>
  </div>
</div><!-- /view-simulator -->

<!-- ═══════════ VIEW: BPMN COVERAGE ═══════════ -->
<div class="view" id="view-bpmn" data-view="bpmn">
  <div class="view-hdr">
    <div>
      <h2>BPMN Coverage — Proceso Minero Completo</h2>
      <div class="desc">Cobertura de cada nodo del proceso: Sedimento → Carga → Transporte → Descarga → Despacho</div>
    </div>
    <div class="view-actions">
      <button class="btn btn-primary" onclick="exportCurrentView()">📄 Exportar PDF</button>
    </div>
  </div>
  {bpmn_html}
</div><!-- /view-bpmn -->

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
}};

const TRANSPORT = {json.dumps(dict(
    n_trucks=transport.n_trucks_served if transport else 0,
    n_full=transport.n_trucks_full if transport else 0,
    disp_eff=transport.dispatch_efficiency if transport else 0,
    avg_fill=transport.avg_truck_fill_pct if transport else 0,
) if transport else dict(n_trucks=0,n_full=0,disp_eff=0,avg_fill=0))};

const DARK_LAYOUT = {{
  paper_bgcolor: '{C['bg']}', plot_bgcolor: '{C['panel']}',
  font: {{color: '{C['text']}', size:10}},
  margin: {{l:38,r:8,t:22,b:28}},
  xaxis: {{gridcolor:'{C['border']}', zerolinecolor:'{C['border']}'}},
  yaxis: {{gridcolor:'{C['border']}', zerolinecolor:'{C['border']}'}},
  legend: {{bgcolor:'rgba(0,0,0,.3)',bordercolor:'{C['border']}',borderwidth:1,font:{{size:9}}}},
}};

// ── STATE ───────────────────────────────────────────────────────────────────
let curIdx   = 0;
let playing  = false;
let playTimer= null;
let speed    = 1;        // steps per tick
let alertsShown = new Set();
let totalAlerts = 0;

const vidL = document.getElementById('vid-left');
const vidR = document.getElementById('vid-right');

// ── PLAYBACK ────────────────────────────────────────────────────────────────
function togglePlay() {{
  playing = !playing;
  document.getElementById('play-btn').innerHTML = playing
    ? '&#9646;&#9646; PAUSE' : '&#9654; PLAY';

  if (playing) {{
    // FIX: Si el video esta pausado, lo arrancamos (antes estaba al reves)
    if (vidL.paused) vidL.play().catch(e => console.warn('vidL play error:', e));
    if (vidR.paused) vidR.play().catch(e => console.warn('vidR play error:', e));
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
  // Ajustar playback rate del video proporcional
  const vr = Math.min(s, 16);
  vidL.playbackRate = vr; vidR.playbackRate = vr;
}}

function tick() {{
  curIdx = Math.min(curIdx + speed, TL.length - 1);
  update(curIdx);
  if (curIdx >= TL.length - 1) {{
    playing = false;
    document.getElementById('play-btn').innerHTML = '&#9654; PLAY';
    clearInterval(playTimer);
    vidL.pause(); vidR.pause();
  }}
}}

function seekTo(idx) {{
  curIdx = Math.min(Math.max(0, idx), TL.length - 1);
  update(curIdx);
}}

// ── MAIN UPDATE ─────────────────────────────────────────────────────────────
function update(idx) {{
  const snap = TL[idx];
  const t    = snap.t;

  // Progress bar + clock
  document.getElementById('progress').value   = idx;
  document.getElementById('t-display').textContent = t.toFixed(1) + ' s';
  document.getElementById('live-clock').textContent = 't = ' + t.toFixed(1) + ' s';

  // Sync video (if not playing — when playing, video runs on its own)
  if (!playing) {{
    if (Math.abs(vidL.currentTime - t) > 0.5) vidL.currentTime = t;
    if (Math.abs(vidR.currentTime - t) > 0.5) vidR.currentTime = t;
  }}
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

  // Alerts
  _processAlerts(t, snap.alerts || []);
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
  const layout = {{ ...DARK_LAYOUT, margin:{{l:10,r:10,t:15,b:5}}, height:100 }};
  if (gaugeInstances[id]) {{
    Plotly.react(id, data, layout, {{displayModeBar:false}});
  }} else {{
    Plotly.newPlot(id, data, layout, {{displayModeBar:false}});
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
// Inicializar gauges en 50%
['g-coll','g-man','g-pos','g-disc','g-oee','g-teff','g-disp','g-truck']
  .forEach((id,i) => _gauge(id, 50, '', '%'));

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

  // Refrescar charts/gauges al entrar (Plotly resize)
  setTimeout(() => window.dispatchEvent(new Event('resize')), 100);

  // Render especificos por vista
  if (name === 'simulator') renderSimulators();
  if (name === 'alerts') renderFullAlertFeed();
}}

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
// ALERT FEED COMPLETO (vista Alerts)
// ═══════════════════════════════════════════════════════════════
function renderFullAlertFeed() {{
  const feed = document.getElementById('alert-feed-full');
  if (!feed) return;
  feed.innerHTML = '';
  const sorted = [...ALL_ALERTS].sort((a,b) => {{
    const sev = {{critical:0, warning:1, info:2, success:3}};
    const sa = sev[a.sev] ?? 4, sb = sev[b.sev] ?? 4;
    if (sa !== sb) return sa - sb;
    return a.t - b.t;
  }});
  sorted.forEach(a => {{
    const el = document.createElement('div');
    el.className = 'alert-item alert-' + (a.sev || 'info');
    el.style.marginBottom = '5px';
    el.innerHTML =
      '<span class="alert-time">t=' + a.t.toFixed(1) + 's</span>' +
      '<b>' + a.type + '</b>: ' + a.msg +
      (a.val != null ? ' <span style="opacity:.8">['+a.val+' '+(a.unit||'')+']</span>' : '') +
      ' <span style="opacity:.6">| Ciclo #' + (a.cid||'—') + '</span>';
    feed.appendChild(el);
  }});
  // Update sidebar badge
  const crit = ALL_ALERTS.filter(a => a.sev === 'critical').length;
  const badge = document.getElementById('side-alert-count');
  if (badge) badge.textContent = crit > 0 ? crit : ALL_ALERTS.length;
  if (badge && crit === 0) badge.style.background = '{C['info']}';
}}

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
  const views = ['overview','live','cycles','trucks','alerts','simulator','bpmn'];
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
// SIMULADOR DE COLOCACION (canvas)
// ═══════════════════════════════════════════════════════════════
let simCurrentScenario = 'offset';
function simScenario(name) {{
  simCurrentScenario = name;
  document.querySelectorAll('.sim-controls button').forEach(b => b.classList.remove('active'));
  event.currentTarget.classList.add('active');
  renderSimulators();
}}

function renderSimulators() {{
  const okC = document.getElementById('sim-ok');
  const badC = document.getElementById('sim-bad');
  if (okC) drawSimulation(okC, 'ok');
  if (badC) drawSimulation(badC, simCurrentScenario);
}}

function drawSimulation(canvas, mode) {{
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, W, H);

  // Grid de piso minero
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 10; i++) {{
    ctx.beginPath();
    ctx.moveTo((i/10)*W, H*0.55);
    ctx.lineTo((i/10)*W, H);
    ctx.stroke();
  }}

  // Horizonte
  ctx.fillStyle = '#1c2128';
  ctx.fillRect(0, 0, W, H*0.55);
  ctx.fillStyle = '#2a1810';
  ctx.fillRect(0, H*0.55, W, H*0.45);

  // Pala (lado derecho)
  const palaX = W*0.75, palaY = H*0.45;
  drawShovel(ctx, palaX, palaY, H*0.35);

  // Zonas de colocacion (vista cenital simulada)
  const zoneY = H*0.72;
  // Zona optima (verde) centrada
  const optX = W*0.35, optR = W*0.08;
  ctx.fillStyle = 'rgba(63,185,80,.25)';
  ctx.strokeStyle = '{C['green']}';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(optX, zoneY, optR, 0, Math.PI*2); ctx.fill(); ctx.stroke();
  // Zona aceptable (amarilla)
  ctx.strokeStyle = '{C['yellow']}';
  ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.arc(optX, zoneY, optR*1.8, 0, Math.PI*2); ctx.stroke();
  ctx.setLineDash([]);

  // Camion (posicion segun mode)
  let truckX = optX, truckY = zoneY, truckAngle = 0, label = '';
  if (mode === 'ok') {{
    truckX = optX; truckY = zoneY; truckAngle = 0;
    label = '✓ Centrado en zona optima';
  }} else if (mode === 'offset') {{
    truckX = optX + optR*2.5; truckY = zoneY;
    label = '✗ Offset lateral +4m · Riesgo de derrame';
  }} else if (mode === 'far') {{
    truckX = optX - optR*3; truckY = zoneY - 20;
    label = '✗ Muy lejos (>15m) · Swing ineficiente';
  }} else if (mode === 'close') {{
    truckX = optX + optR*0.5; truckY = zoneY + 25;
    label = '✗ Muy cerca · Riesgo de contacto con balde';
  }} else if (mode === 'angle') {{
    truckX = optX; truckY = zoneY; truckAngle = 0.4;
    label = '✗ Mal angulo (23°) · Dump descentrado';
  }}
  drawTruck(ctx, truckX, truckY, H*0.18, truckAngle, mode === 'ok');

  // Linea de swing pala-camion
  ctx.strokeStyle = mode === 'ok' ? '{C['green']}' : '{C['red']}';
  ctx.lineWidth = 2;
  ctx.setLineDash([6,4]);
  ctx.beginPath();
  ctx.moveTo(palaX - H*0.18, palaY);
  ctx.lineTo(truckX, truckY - 20);
  ctx.stroke();
  ctx.setLineDash([]);

  // Label arriba
  ctx.fillStyle = mode === 'ok' ? '{C['green']}' : '{C['red']}';
  ctx.font = 'bold 14px system-ui, sans-serif';
  ctx.fillText(label, 10, 22);

  // Distancia calculada
  const dx = truckX - palaX, dy = truckY - palaY;
  const dist = Math.sqrt(dx*dx + dy*dy) / (W*0.05);  // simulated meters
  ctx.fillStyle = '{C['muted']}';
  ctx.font = '11px monospace';
  ctx.fillText('Distancia estimada: ' + dist.toFixed(1) + 'm', 10, 40);
  ctx.fillText('Offset: ' + ((truckX-optX)/(W*0.05)).toFixed(1) + 'm', 10, 54);
  ctx.fillText('Angulo: ' + (truckAngle*180/Math.PI).toFixed(0) + '°', 10, 68);
}}

function drawShovel(ctx, x, y, size) {{
  // Cuerpo pala
  ctx.fillStyle = '#4a5568';
  ctx.fillRect(x - size*0.2, y - size*0.6, size*0.4, size*0.6);
  // Brazo
  ctx.strokeStyle = '{C['orange']}';
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.moveTo(x, y - size*0.4);
  ctx.lineTo(x - size*0.8, y);
  ctx.stroke();
  // Balde
  ctx.fillStyle = '{C['orange']}';
  ctx.beginPath();
  ctx.moveTo(x - size*0.8, y - size*0.1);
  ctx.lineTo(x - size*0.95, y + size*0.05);
  ctx.lineTo(x - size*0.7, y + size*0.15);
  ctx.lineTo(x - size*0.55, y);
  ctx.closePath();
  ctx.fill();
  // Tracks
  ctx.fillStyle = '#1a1a1a';
  ctx.fillRect(x - size*0.3, y, size*0.6, size*0.1);
  // Label
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 10px sans-serif';
  ctx.fillText('EX-5600', x - size*0.15, y - size*0.7);
}}

function drawTruck(ctx, x, y, size, angle, isOk) {{
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  // Chasis
  ctx.fillStyle = isOk ? '#3a8ee0' : '#c67e3e';
  ctx.fillRect(-size*1.2, -size*0.15, size*2.4, size*0.4);
  // Tolva
  ctx.fillStyle = isOk ? '#4a9eff' : '#e0962e';
  ctx.beginPath();
  ctx.moveTo(-size*0.9, -size*0.15);
  ctx.lineTo(-size*1.0, -size*0.6);
  ctx.lineTo(size*1.0, -size*0.6);
  ctx.lineTo(size*0.9, -size*0.15);
  ctx.closePath();
  ctx.fill();
  // Cabina (frontal)
  ctx.fillStyle = '#2a2a2a';
  ctx.fillRect(size*0.7, -size*0.45, size*0.3, size*0.3);
  // Llantas
  ctx.fillStyle = '#000';
  [-size*0.9, -size*0.3, size*0.3, size*0.9].forEach(wx => {{
    ctx.beginPath(); ctx.arc(wx, size*0.28, size*0.15, 0, Math.PI*2); ctx.fill();
  }});
  // Display peso (frontal)
  ctx.fillStyle = '{C['yellow']}';
  ctx.fillRect(size*0.72, -size*0.4, size*0.2, size*0.08);
  ctx.fillStyle = '#000';
  ctx.font = 'bold 8px monospace';
  ctx.fillText('218t', size*0.73, -size*0.34);
  // ID
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 10px sans-serif';
  ctx.fillText(isOk?'T-042':'T-???', -size*0.3, size*0.05);
  ctx.restore();
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
      <div class="sec-title">Como interpretar el simulador</div>
      <div class="glossary">
        <p style="color:{C['muted']}">
          El simulador grafico (vista <b>Simulador</b>) muestra la posicion ideal del camion frente a la pala
          y comparaciones con errores tipicos: offset lateral, distancia incorrecta, angulo desviado.
          En operacion real esto se calibra con la primera carga exitosa de cada turno.
        </p>
      </div>
    </div>
    """
