@echo off
REM ============================================================================
REM run_classify.bat — Compile and run the event-driven SNN digit classifier
REM ============================================================================

set VIVADO_BIN=C:\Xilinx\Vivado\2020.1\bin
set PATH=%VIVADO_BIN%;%PATH%
cd /d "%~dp0"

echo ============================================================
echo  Compiling Event-Driven SNN Classifier...
echo ============================================================
call xvlog --sv -i . ..\rtl\neuron_core.sv ..\rtl\synapse_csr.sv ..\rtl\spike_router.sv ..\rtl\top.sv ..\tb\tb_classify.sv
if errorlevel 1 (
    echo [ERROR] Compilation failed!
    exit /b 1
)

echo.
echo  Elaborating...
call xelab tb_classify -debug typical -s sim_classify
if errorlevel 1 (
    echo [ERROR] Elaboration failed!
    exit /b 1
)

echo.
echo ============================================================
echo  Running event-driven classification...
echo ============================================================
call xsim sim_classify -runall
if errorlevel 1 (
    echo [ERROR] Simulation failed!
    exit /b 1
)
