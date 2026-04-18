# -*- coding: utf-8 -*-
"""pipeline.py — Orquestador principal JEBI 2026"""

import os, sys, json, argparse, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from imu_processor import load_imu, detect_cycles, cycles_to_dataframe
from video_processor import process_video_events
from metrics import (compute_efficiency_profiles, compute_truck_loads,
                     compute_transport_metrics, build_realtime_timeline,
                     compute_wear_score, extract_spill_events)
from reporter import generate_report

INPUT_FILES = {
    'left':  ['shovel_left.mp4',
              '40343737_20260313_110600_to_112100_left.mp4'],
    'right': ['shovel_right.mp4',
              '40343737_20260313_110600_to_112100_right.mp4'],
    'imu':   ['imu_data.csv', 'imu_data.npy',
              '40343737_20260313_110600_to_112100_imu.npy'],
}

def find_input(d, candidates):
    for n in candidates:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    return None


def run(inputs_dir, outputs_dir):
    t0 = time.time()
    os.makedirs(outputs_dir, exist_ok=True)

    print('='*60)
    print('  JEBI 2026  Shovel Intelligence Pipeline v2')
    print('='*60)

    # ── 1. IMU ────────────────────────────────────────────────────────────────
    imu_path = find_input(inputs_dir, INPUT_FILES['imu'])
    if not imu_path:
        print('ERROR: imu_data.csv no encontrado en inputs/'); sys.exit(1)

    print(f'\n[1/6] IMU: {os.path.basename(imu_path)}')
    df = load_imu(imu_path)

    # ── 2. Ciclos ─────────────────────────────────────────────────────────────
    print('\n[2/6] Detectando ciclos, mini-ciclos, wait events...')
    cycles, waits, metrics = detect_cycles(df)
    full = [c for c in cycles if not c.is_mini_cycle]
    mini = [c for c in cycles if c.is_mini_cycle]
    print(f'  Ciclos completos : {len(full)}')
    print(f'  Mini-ciclos      : {len(mini)}')
    print(f'  Wait events      : {len(waits)}')

    # ── 3. Video (backtracking + YOLO26 + OCR) ────────────────────────────────
    left_path  = find_input(inputs_dir, INPUT_FILES['left'])
    right_path = find_input(inputs_dir, INPUT_FILES['right'])
    video_events: list = []
    ocr_events:   list = []
    dust_index:   dict = {}
    if left_path and right_path:
        print('\n[3/6] Backtracking video + YOLO26 + OCR...')
        video_result = process_video_events(left_path, right_path, cycles, waits)
        video_events = video_result.get('events', [])
        ocr_events   = video_result.get('ocr_events', [])
        dust_index   = video_result.get('dust_index', {})
        print(f'  Eventos video : {len(video_events)}')
        print(f'  OCR readings  : {len(ocr_events)}')
        print(f'  Dust index avg: {dust_index.get("avg", 0):.1f}%')
    else:
        print('\n[3/6] Video no disponible')

    # ── 4. Eficiencias y alertas ─────────────────────────────────────────────
    print('\n[4/6] Calculando eficiencias, alertas, wear y desperdicios...')
    profiles, alerts = compute_efficiency_profiles(cycles, waits, df)
    truck_events     = compute_truck_loads(cycles, video_events)
    transport        = compute_transport_metrics(truck_events, waits, df.attrs['duration_s'])
    wear_score       = compute_wear_score(alerts, df, cycles)
    spill_events     = extract_spill_events(alerts)
    print(f'  Alertas generadas: {len(alerts)}')
    print(f'  Camiones trackados: {transport.n_trucks_served}')
    print(f'  Wear score total: {wear_score.get("total_score", 0):.1f}')
    print(f'  Spill events     : {len(spill_events)}')

    # ── 5. Timeline real-time ─────────────────────────────────────────────────
    print('\n[5/6] Construyendo timeline real-time...')
    timeline = build_realtime_timeline(df, cycles, waits, profiles, alerts, step_s=0.5)
    print(f'  Snapshots: {len(timeline)} (cada 0.5s)')

    # ── 6. Outputs ───────────────────────────────────────────────────────────
    print('\n[6/6] Generando outputs...')
    df_cycles, df_waits = cycles_to_dataframe(cycles, waits)
    df_cycles.to_csv(os.path.join(outputs_dir, 'cycles.csv'), index=False)
    df_waits.to_csv( os.path.join(outputs_dir, 'waits.csv'),  index=False)

    # Alertas CSV
    import pandas as pd
    pd.DataFrame([dict(
        t=a.timestamp_s, cycle_id=a.cycle_id, type=a.alert_type,
        severity=a.severity, message=a.message,
        value=a.value, unit=a.unit
    ) for a in alerts]).to_csv(os.path.join(outputs_dir,'alerts.csv'), index=False)

    # JSON completo
    report = dict(
        session=metrics.__dict__,
        transport=transport.__dict__,
        cycles=[dict(
            cycle_id=c.cycle_id, type='mini' if c.is_mini_cycle else 'full',
            t_start=round(c.t_start,2), t_end=round(c.t_end,2),
            t_dump=round(c.t_dump,2) if c.t_dump else None,
            duration=round(c.t_end-c.t_start,1),
            fill_pct=round(c.fill_factor*100,1),
            payload_t=round(c.payload_t,1),
            truck_id=c.truck_id or 'unknown',
        ) for c in cycles],
        wait_events=[dict(
            t_start=round(w.t_start,2), t_end=round(w.t_end,2),
            duration=round(w.duration_s,1), reason=w.reason
        ) for w in waits],
        alerts=[dict(
            t=a.timestamp_s, type=a.alert_type, severity=a.severity,
            message=a.message, value=a.value, cycle_id=a.cycle_id
        ) for a in alerts],
    )
    with open(os.path.join(outputs_dir,'report.json'),'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Dashboard
    html_path = generate_report(
        df_imu=df, cycles=cycles, wait_events=waits, metrics=metrics,
        video_events=video_events, df_cycles=df_cycles, df_waits=df_waits,
        output_dir=outputs_dir,
        profiles=profiles, alerts=alerts,
        truck_events=truck_events, transport=transport,
        timeline=timeline,
        ocr_events=ocr_events, wear_score=wear_score,
        spill_events=spill_events, dust_index=dust_index,
    )

    # OCR readings CSV
    if ocr_events:
        pd.DataFrame([dict(
            t=ev.get('timestamp_s', 0),
            cycle_id=ev.get('cycle_id', 0),
            truck_id=ev.get('truck_id_ocr', 'unknown'),
            conf_id=round(ev.get('confidence_id', 0), 2),
            weight=ev.get('weight_reading', '—'),
            weight_unit=ev.get('weight_unit', ''),
            conf_weight=round(ev.get('confidence_weight', 0), 2),
            model=ev.get('truck_model', 'unknown'),
            position_score=ev.get('position_score', 0),
            method=ev.get('method', 'none'),
        ) for ev in ocr_events]).to_csv(
            os.path.join(outputs_dir, 'ocr_readings.csv'), index=False)

    print(f'\n{"="*60}')
    print(f'  Listo en {time.time()-t0:.1f}s')
    print(f'  Dashboard : {html_path}')
    print(f'  Alertas   : {len(alerts)} eventos')
    print(f'  Ciclos    : {len(full)} completos + {len(mini)} mini')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs',  default='./inputs')
    ap.add_argument('--outputs', default='./outputs')
    args = ap.parse_args()
    base = Path(__file__).resolve().parent.parent
    inp = str((base / args.inputs).resolve())
    out = str((base / args.outputs).resolve())
    run(inp, out)
