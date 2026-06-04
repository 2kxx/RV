#!/usr/bin/env bash
# scripts/v_arr_mvp/setup_baseline_envs.sh
# ----------------------------------------------------------------------------
# Install missing dependencies for VideoScore-v1.1 and VideoScore2 baselines
# into the CURRENT active Python environment (rewardverse).
#
# Dependency audit (2026-05-26):
#
#   Package          rewardverse   VS-v1.1 needs   VS2 needs
#   -------          -----------   --------------   ---------
#   torch            2.6.0+cu124   any              2.6.0  ✅
#   transformers     4.53.2        mantis-vl only   4.53.2 ✅
#   qwen_vl_utils    0.0.14        -                any    ✅
#   decord           0.6.0         any              any    ✅
#   accelerate       1.13.0        -                any    ✅
#   numpy            2.2.6         -                2.2.6  ✅
#   scipy            1.15.3        -                any    ✅
#   opencv-python    ?             -                any    (check below)
#   mantis-vl        MISSING       REQUIRED         -
#
# mantis-vl provides Idefics2ForSequenceClassification used by VS-v1.1.
# baseline_videoscore.py already contains compat shims for
# mantis-vl <-> transformers >= 4.50/4.53/4.56, so no version downgrade needed.
#
# Usage:
#   bash scripts/v_arr_mvp/setup_baseline_envs.sh
# ----------------------------------------------------------------------------
set -uo pipefail

PYTHON="${PYTHON:-python}"

echo "============================================================"
echo "[setup] Installing baseline dependencies into: $(${PYTHON} -c 'import sys; print(sys.prefix)')"
echo "============================================================"

# ---- 1. mantis-vl (VideoScore-v1.1) ----
echo ""
echo "[1/2] mantis-vl (required by VideoScore-v1.1)..."
if ${PYTHON} -c "from mantis.models.idefics2 import Idefics2ForSequenceClassification" 2>/dev/null; then
    echo "  [skip] mantis-vl already installed."
else
    echo "  [install] pip install mantis-vl ..."
    pip install mantis-vl --quiet
    echo "  [verify]"
    ${PYTHON} -c "from mantis.models.idefics2 import Idefics2ForSequenceClassification; print('  mantis-vl: OK')"
fi

# ---- 2. opencv-python-headless (VideoScore2) ----
echo ""
echo "[2/2] opencv-python-headless (required by VideoScore2)..."
if ${PYTHON} -c "import cv2" 2>/dev/null; then
    echo "  [skip] opencv already installed: $(${PYTHON} -c 'import cv2; print(cv2.__version__)')"
else
    echo "  [install] pip install opencv-python-headless ..."
    pip install opencv-python-headless --quiet
    echo "  [verify]"
    ${PYTHON} -c "import cv2; print('  opencv:', cv2.__version__)"
fi

# ---- Final verification ----
echo ""
echo "============================================================"
echo "[setup] Final package check:"
${PYTHON} -c "
pkgs = [
    ('torch',           'torch'),
    ('transformers',    'transformers'),
    ('decord',          'decord'),
    ('qwen_vl_utils',   'qwen_vl_utils'),
    ('accelerate',      'accelerate'),
    ('numpy',           'numpy'),
    ('scipy',           'scipy'),
    ('cv2',             'opencv-python-headless'),
    ('mantis.models.idefics2', 'mantis-vl'),
]
all_ok = True
for mod, label in pkgs:
    try:
        m = __import__(mod.split('.')[0])
        ver = getattr(m, '__version__', '?')
        print(f'  ✅  {label:<28} {ver}')
    except ImportError:
        print(f'  ❌  {label:<28} NOT FOUND')
        all_ok = False
if all_ok:
    print()
    print('  All dependencies satisfied. Ready to run baselines.')
else:
    import sys; sys.exit(1)
"
echo "============================================================"
