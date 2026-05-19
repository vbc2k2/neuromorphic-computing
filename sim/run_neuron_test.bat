@echo off
REM ============================================================================
REM run_neuron_test.bat — Compile and run neuron core unit test
REM ============================================================================

set VIVADO_BIN=C:\Xilinx\Vivado\2020.1\bin
set PATH=%VIVADO_BIN%;%PATH%
cd /d "%~dp0"

echo ============================================================
echo  Neuron Core Unit Test
echo ============================================================

call xvlog --sv ..\rtl\neuron_core.sv ..\tb\tb_neuron_core.sv
if errorlevel 1 (
    echo [ERROR] Compilation failed!
    exit /b 1
)

call xelab tb_neuron_core -debug typical -s sim_neuron
if errorlevel 1 (
    echo [ERROR] Elaboration failed!
    exit /b 1
)

call xsim sim_neuron -runall

echo.
echo  Done.
echo ============================================================
