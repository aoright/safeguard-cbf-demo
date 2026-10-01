#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  SafeGuard CBF-QP Zero-Dependency Evaluation"
echo "============================================================"

# 1. Environment check or setup
if [ -d "venv" ]; then
    echo "Using existing virtual environment 'venv'..."
    source venv/bin/activate
elif command -v python3 &> /dev/null; then
    echo "Checking Python environment..."
else
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Ensure dependencies are installed
if ! python3 -c "import numpy, scipy, matplotlib, cvxpy" 2>/dev/null; then
    echo "Installing missing dependencies from requirements.txt..."
    pip install -r requirements.txt
fi

echo ""
echo "--- Running Unit Tests ---"
python3 -m unittest discover -s tests -p "test_*.py" -v

echo ""
echo "--- Running CBF-QP Simulation Demo ---"
python3 safeguard_cbf_demo.py

echo ""
echo "============================================================"
echo "  Evaluation completed successfully!"
echo "============================================================"
