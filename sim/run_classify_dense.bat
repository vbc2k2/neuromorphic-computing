@echo off
REM ============================================================================
REM run_classify_dense.bat — Compile and run the dense-baseline SNN classifier
REM ============================================================================

set VIVADO_BIN=C:\Xilinx\Vivado\2020.1\bin
set PATH=%VIVADO_BIN%;%PATH%
cd /d "%~dp0"

echo ============================================================
echo  Compiling Dense-Baseline SNN Classifier...
echo ============================================================
call xvlog --sv -i . ..\rtl\neuron_core.sv ..\rtl\synapse_csr.sv ..\rtl\top_dense.sv ..\tb\tb_classify_dense.sv
if errorlevel 1 (
    echo [ERROR] Compilation failed!
    exit /b 1
)

echo.
echo  Elaborating...
call xelab tb_classify_dense -debug typical -s sim_dense
if errorlevel 1 (
    echo [ERROR] Elaboration failed!
    exit /b 1
)

echo.
echo ============================================================
echo  Running dense-baseline classification...
echo ============================================================
call xsim sim_dense -runall
if errorlevel 1 (
    echo [ERROR] Simulation failed!
    exit /b 1
)
