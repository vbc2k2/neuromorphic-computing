@echo off
REM ============================================================================
REM run_all.bat — Full SNN accelerator benchmark flow:
REM   train -> unit test -> event classifier -> dense baseline -> compare
REM ============================================================================

set VIVADO_BIN=C:\Xilinx\Vivado\2020.1\bin
set PATH=%VIVADO_BIN%;%PATH%
cd /d "%~dp0"

echo ============================================================
echo  SNN Accelerator Benchmark - full flow
echo ============================================================

echo.
echo [1/5] Train SNN + export weights / spikes / config
python ..\python\train_snn.py
if errorlevel 1 goto :error

echo.
echo [2/5] Neuron core unit test
call "%~dp0run_neuron_test.bat"
if errorlevel 1 goto :error

echo.
echo [3/5] Event-driven SNN classifier
call "%~dp0run_classify.bat"
if errorlevel 1 goto :error

echo.
echo [4/5] Dense-baseline SNN classifier
call "%~dp0run_classify_dense.bat"
if errorlevel 1 goto :error

echo.
echo [5/5] Compare accuracy + efficiency
python ..\python\compare_classify.py
if errorlevel 1 goto :error

echo.
echo ============================================================
echo  All done! See results/ for the comparison plot and CSV.
echo ============================================================
goto :eof

:error
echo.
echo [ERROR] A step failed. Aborting.
exit /b 1
