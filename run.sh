#!/bin/bash
# JEBI Hackathon 2026 - Grupo 05
# Shovel Intelligence: Cycle Detection + Payload Estimation + YOLO26 + OCR + Dashboard

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUTS="$SCRIPT_DIR/inputs"
OUTPUTS="$SCRIPT_DIR/outputs"

echo "======================================================"
echo "  JEBI 2026 - Grupo 05 - Shovel Intelligence v3"
echo "  Pipeline: IMU + Video (YOLO26) + OCR + Dashboard"
echo "======================================================"
echo "Inputs:"
ls "$INPUTS/" 2>/dev/null || echo "  (directorio vacio)"

mkdir -p "$OUTPUTS"

# ── Dependencias core (fallan rapido si no se pueden instalar) ──────────────
echo ""
echo "Instalando dependencias core..."
pip install -q --no-warn-script-location \
    "opencv-python>=4.8.0" \
    "numpy>=1.24.0" \
    "pandas>=2.0.0" \
    "scipy>=1.11.0" || true

# ── EasyOCR (opcional pero recomendado para OCR) ────────────────────────────
pip install -q easyocr>=1.7.0 2>/dev/null || \
    echo "  easyocr no disponible - OCR deshabilitado"

# ── Ultralytics YOLO26 (opcional, el pipeline tiene fallback) ───────────────
# Se usa para deteccion precisa de camiones en los frames
pip install -q "ultralytics>=8.3.0" 2>/dev/null || \
    echo "  ultralytics no disponible - usando detector visual legacy"

# Correr pipeline principal
echo ""
echo "Corriendo pipeline..."
python "$SCRIPT_DIR/solution/pipeline.py" \
    --inputs  "$INPUTS" \
    --outputs "$OUTPUTS"

echo ""
echo "Outputs generados:"
ls -la "$OUTPUTS/"
echo ""
echo "Dashboard interactivo: $OUTPUTS/dashboard.html"
echo "Done."
