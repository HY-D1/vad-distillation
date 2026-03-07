@echo off
:: start.bat - Windows entry point for start.ps1
::
:: Usage: start.bat <MODE> [OPTIONS]
::
:: This batch file simply forwards to the PowerShell script.
:: For full help, run: start.bat help

powershell -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
