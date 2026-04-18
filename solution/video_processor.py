# -*- coding: utf-8 -*-
"""
video_processor.py
Backtracking sobre video: solo va a los timestamps relevantes (dump events).
Extrae frames clave, detecta truck ID por OCR y forma visual.
"""

import cv2
import numpy as np
import base64
import re
import os
from typing import List, Dict, Optional

# Intenta importar easyocr — si no esta, usa fallback
try:
    import easyocr
    _OCR_READER = None   # lazy init
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Vision detector avanzado (YOLO26 + OCR especializado)
try:
    from vision_detector import analyze_frame, compute_dust_index
    VISION_V2_AVAILABLE = True
except ImportError:
    VISION_V2_AVAILABLE = False
    print('  [video] vision_detector no disponible, usando metodo legacy')

# ─── CONFIGURACION ────────────────────────────────────────────────────────────

FRAME_THUMB_W  = 480
FRAME_THUMB_H  = 270
TRUCK_NUMBERS_PATTERN = re.compile(r'\b\d{2,5}\b')  # 2-5 digitos

# Capacidades conocidas por modelo visual
TRUCK_CAPS = {
    'CAT_793F':   221.0,
    'EH4000_AC3': 218.0,
    'unknown':    219.0,   # promedio si no detectamos
}


# ─── BACKTRACKING ─────────────────────────────────────────────────────────────

def extract_frames_at_timestamps(
    video_path: str,
    timestamps_s: List[float],
    offset_s: float = -0.5
) -> Dict[float, np.ndarray]:
    """
    BACKTRACKING: usa random access de OpenCV para ir directo
    a cada timestamp sin escanear el video completo.
    
    offset_s: captura el frame un poco antes del dump (mejor vista del camion)
    Retorna dict {timestamp: frame_bgr}
    """
    if not os.path.exists(video_path):
        print(f"  [video] Archivo no encontrado: {video_path}")
        return {}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 15.0

    frames = {}
    for ts in timestamps_s:
        seek_ms = max(0.0, (ts + offset_s) * 1000.0)
        cap.set(cv2.CAP_PROP_POS_MSEC, seek_ms)
        ret, frame = cap.read()
        if ret:
            frames[ts] = frame

    cap.release()
    print(f"  [video] Extraidos {len(frames)}/{len(timestamps_s)} frames por backtracking")
    return frames


def extract_stereo_frames(
    left_path: str,
    right_path: str,
    timestamps_s: List[float],
    offset_s: float = -0.5
) -> Dict[float, Dict[str, np.ndarray]]:
    """
    Extrae frames de ambas camaras para los mismos timestamps.
    Retorna { timestamp: {'left': frame, 'right': frame} }
    """
    left_frames  = extract_frames_at_timestamps(left_path,  timestamps_s, offset_s)
    right_frames = extract_frames_at_timestamps(right_path, timestamps_s, offset_s)

    result = {}
    for ts in timestamps_s:
        result[ts] = {
            'left':  left_frames.get(ts),
            'right': right_frames.get(ts),
        }
    return result


# ─── DETECCION DE CAMION ──────────────────────────────────────────────────────

def detect_truck(frame: np.ndarray) -> Dict:
    """
    Detecta truck_id y modelo en un frame.
    Combina:
      1. OCR del numero pintado en el camion
      2. Clasificacion visual por forma (aspect ratio del area del camion)
    
    Retorna {
      'truck_id': str,
      'model': str,
      'capacity_t': float,
      'confidence': float,
      'method': str
    }
    """
    result = {
        'truck_id':   'unknown',
        'model':      'unknown',
        'capacity_t': TRUCK_CAPS['unknown'],
        'confidence': 0.0,
        'method':     'none',
    }

    if frame is None:
        return result

    # ── Intento 1: OCR ────────────────────────────────────────────────────────
    if OCR_AVAILABLE:
        ocr_result = _ocr_truck_number(frame)
        if ocr_result:
            result.update({
                'truck_id':   ocr_result,
                'confidence': 0.75,
                'method':     'ocr',
            })
            # Asignar modelo/capacidad por rango de ID (heuristica mineria)
            result['model']      = _infer_model_from_id(ocr_result)
            result['capacity_t'] = TRUCK_CAPS.get(result['model'], TRUCK_CAPS['unknown'])
            return result

    # ── Intento 2: Clasificacion por silueta ──────────────────────────────────
    model, conf = _classify_truck_visual(frame)
    if conf > 0.4:
        result.update({
            'model':      model,
            'capacity_t': TRUCK_CAPS.get(model, TRUCK_CAPS['unknown']),
            'confidence': conf,
            'method':     'visual',
            'truck_id':   f'truck_{model}',
        })

    return result


def _ocr_truck_number(frame: np.ndarray) -> Optional[str]:
    """OCR sobre frame — busca numeros de 2-5 digitos (ID de camion)."""
    global _OCR_READER
    try:
        if _OCR_READER is None:
            _OCR_READER = easyocr.Reader(['en'], verbose=False)

        # ROI: parte inferior del frame (donde aparecen los numeros del camion)
        h, w = frame.shape[:2]
        roi = frame[h//3:, :]   # mitad inferior

        results = _OCR_READER.readtext(roi, detail=1)
        candidates = []
        for (bbox, text, conf) in results:
            cleaned = text.strip().upper()
            nums = TRUCK_NUMBERS_PATTERN.findall(cleaned)
            if nums and conf > 0.5:
                candidates.append((conf, nums[0]))

        if candidates:
            best = sorted(candidates, reverse=True)[0]
            return best[1]
    except Exception as e:
        print(f"  [OCR] Error: {e}")
    return None


def _classify_truck_visual(frame: np.ndarray) -> tuple:
    """
    Clasificacion visual simple CAT 793F vs EH4000 AC-3
    basada en deteccion de silueta grande en la imagen.
    
    CAT 793F:   aspect ratio caja de carga ~ 1.6-1.8
    EH4000 AC3: aspect ratio caja de carga ~ 1.3-1.5  (mas alto/ancho)
    """
    if frame is None:
        return 'unknown', 0.0

    try:
        # Convertir a gris y detectar contornos grandes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Buscar el objeto mas grande en la imagen (el camion)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Tomar el contorno mas grande
        large = [c for c in contours if cv2.contourArea(c) > (w * h * 0.02)]
        if not large:
            return 'unknown', 0.1

        biggest = max(large, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(biggest)
        aspect = bw / max(bh, 1)

        # Clasificacion por aspect ratio
        if 1.55 <= aspect <= 1.90:
            return 'CAT_793F', 0.55
        elif 1.25 <= aspect <= 1.54:
            return 'EH4000_AC3', 0.55
        else:
            # Fallback: area relativa da pista del modelo
            fill = cv2.contourArea(biggest) / (w * h)
            if fill > 0.35:
                return 'CAT_793F', 0.40    # camiones mas grandes llenan mas frame
            return 'EH4000_AC3', 0.40

    except Exception:
        return 'unknown', 0.0


def _infer_model_from_id(truck_id_str: str) -> str:
    """
    Heuristica: en muchas minas los IDs de camiones son consecutivos
    y los modelos estan en rangos distintos.
    Sin info de flota real, alternamos conservadoramente.
    """
    try:
        n = int(truck_id_str)
        if n % 2 == 0:
            return 'CAT_793F'
        else:
            return 'EH4000_AC3'
    except Exception:
        return 'unknown'


# ─── THUMBNAILS PARA EL DASHBOARD ─────────────────────────────────────────────

def frame_to_base64(frame: np.ndarray,
                    width: int = FRAME_THUMB_W,
                    height: int = FRAME_THUMB_H,
                    label: str = '') -> str:
    """
    Convierte frame BGR a PNG base64 para incrustar en HTML.
    Opcional: agrega label en la esquina superior izquierda.
    """
    if frame is None:
        # Placeholder negro con texto
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(placeholder, 'No frame', (10, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        frame = placeholder

    resized = cv2.resize(frame, (width, height))

    if label:
        cv2.rectangle(resized, (0, 0), (len(label) * 9 + 10, 22), (0, 0, 0), -1)
        cv2.putText(resized, label, (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    _, buf = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode('utf-8')


def build_stereo_thumb(left_frame: Optional[np.ndarray],
                        right_frame: Optional[np.ndarray],
                        label: str = '') -> str:
    """Combina left + right en una sola imagen base64 para el dashboard."""
    h, w = FRAME_THUMB_H, FRAME_THUMB_W

    def safe(f):
        if f is None:
            ph = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(ph, 'no frame', (5, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
            return ph
        return cv2.resize(f, (w, h))

    combined = np.hstack([safe(left_frame), safe(right_frame)])

    if label:
        cv2.rectangle(combined, (0, 0), (len(label)*9 + 10, 22), (0, 0, 0), -1)
        cv2.putText(combined, label, (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    _, buf = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return base64.b64encode(buf).decode('utf-8')


def process_video_events(
    left_path: str,
    right_path: str,
    cycles,         # List[LoadCycle]
    wait_events,    # List[WaitEvent]
) -> Dict:
    """
    Pipeline completo de video:
      1. Recoge timestamps de todos los dumps (backtracking)
      2. Extrae frames de ambas camaras
      3. YOLO26 + OCR (ID + medidor peso) sobre cada dump
      4. Asigna truck_id a cada ciclo
      5. Genera thumbnails + frames anotados para el dashboard
      6. Calcula dust index por frame (para BPMN node Aire)

    Retorna dict con:
      {
        'events':      list legacy (thumb_b64, truck_id, ...)   <- compat
        'ocr_events':  list con OCR completo + frame anotado
        'dust_index':  {'avg':float, 'max':float, 'high_count':int}
      }
    """
    # ── Recolectar timestamps de interes ──────────────────────────────────────
    event_timestamps = {}
    for c in cycles:
        if c.t_dump is not None:
            event_timestamps[c.t_dump] = ('dump', c.cycle_id)
        event_timestamps[c.t_start] = ('start', c.cycle_id)

    ts_list = sorted(event_timestamps.keys())

    if not ts_list:
        return {'events': [], 'ocr_events': [], 'dust_index': {}}

    # ── Backtracking ──────────────────────────────────────────────────────────
    print(f"  [video] Backtracking a {len(ts_list)} timestamps clave...")
    stereo = extract_stereo_frames(left_path, right_path, ts_list, offset_s=-1.0)

    # ── Deteccion YOLO26 + OCR + thumbnails ───────────────────────────────────
    events: List[Dict] = []
    ocr_events: List[Dict] = []
    cycle_truck: Dict[int, Dict] = {}
    dust_scores: List[float] = []

    print('  [video] Corriendo YOLO26 + OCR sobre eventos clave...')
    for ts in ts_list:
        ev_type, cycle_id = event_timestamps[ts]
        left_f  = stereo[ts].get('left')
        right_f = stereo[ts].get('right')

        # ── V2: YOLO26 + OCR completo sobre el frame izquierdo ───────────────
        ocr_data = None
        if VISION_V2_AVAILABLE and left_f is not None:
            try:
                ocr_data = analyze_frame(left_f, ts, cycle_id, ev_type)
            except Exception as e:
                print(f"    [video] analyze_frame fallo en t={ts:.1f}: {e}")
                ocr_data = None

            # Dust index del frame
            try:
                dust_scores.append(compute_dust_index(left_f))
            except Exception:
                pass

        # ── Truck info consolidada (YOLO preferido, fallback legacy) ─────────
        if ocr_data and ocr_data.get('truck_detected'):
            truck_info = {
                'truck_id':   (ocr_data.get('truck_id_ocr', 'unknown')
                               if ocr_data.get('truck_id_ocr') != 'unknown'
                               else f"T-{cycle_id:03d}"),
                'model':      ocr_data.get('truck_model', 'unknown'),
                'capacity_t': ocr_data.get('capacity_t', 219.0),
                'confidence': ocr_data.get('confidence_id', 0.0),
                'method':     'yolo26+ocr',
            }
            # Registrar OCR event completo
            if ev_type == 'dump':
                ocr_events.append(ocr_data)
        else:
            # Fallback visual legacy
            truck_info = detect_truck(left_f) if left_f is not None else {
                'truck_id': 'unknown', 'model': 'unknown',
                'capacity_t': 219.0, 'confidence': 0.0, 'method': 'none',
            }

        # Thumbnail combinado (compatibilidad)
        thumb = build_stereo_thumb(
            left_f, right_f,
            label=f"Ciclo #{cycle_id} | {ev_type.upper()} | t={ts:.1f}s"
        )

        events.append({
            'timestamp_s':  ts,
            'event_type':   ev_type,
            'cycle_id':     cycle_id,
            'truck_id':     truck_info['truck_id'],
            'truck_model':  truck_info['model'],
            'capacity_t':   truck_info['capacity_t'],
            'confidence':   truck_info['confidence'],
            'method':       truck_info['method'],
            'thumb_b64':    thumb,
        })

        if ev_type == 'dump':
            cycle_truck[cycle_id] = truck_info

    # ── Asignar truck_id a ciclos ──────────────────────────────────────────────
    for c in cycles:
        if c.cycle_id in cycle_truck:
            info = cycle_truck[c.cycle_id]
            c.truck_id = info['truck_id']

    # ── Dust index agregado ───────────────────────────────────────────────────
    if dust_scores:
        dust_avg = float(np.mean(dust_scores))
        dust_max = float(np.max(dust_scores))
        high_count = int(sum(1 for d in dust_scores if d > 60))
    else:
        dust_avg, dust_max, high_count = 0.0, 0.0, 0

    print(f"  [video] {len(events)} eventos + {len(ocr_events)} OCR events procesados")

    return {
        'events':     events,
        'ocr_events': ocr_events,
        'dust_index': {
            'avg':        round(dust_avg, 1),
            'max':        round(dust_max, 1),
            'high_count': high_count,
            'n_samples':  len(dust_scores),
        },
    }
