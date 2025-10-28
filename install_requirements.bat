@echo off
echo Installing CCTV System with Person Detection...
echo.

echo Installing PyTorch (CPU version)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installing other requirements...
pip install -r requirements.txt

echo.
echo Installation complete!
echo.
echo To use GPU acceleration (if you have NVIDIA GPU), run:
echo pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
echo.
pause