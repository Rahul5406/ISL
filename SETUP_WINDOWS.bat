@echo off
echo ========================================
echo  ISL Detection - Windows Setup Fix
echo ========================================
echo.
echo [1/2] Downgrading mediapipe to 0.10.14 (required for solutions API)...
pip install mediapipe==0.10.14 --force-reinstall
echo.
echo [2/2] Installing other dependencies...
pip install opencv-python numpy scikit-learn pandas
echo.
echo ========================================
echo  DONE! Now run in this order:
echo    python 1_collect_data.py
echo    python 2_train_model.py
echo    python 3_detect_realtime.py
echo ========================================
pause
