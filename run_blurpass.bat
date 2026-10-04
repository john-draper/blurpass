@echo off
REM BlurPass - scan videos for gore locally and render blurred copies
REM usage: run_blurpass scan "D:\movie.mkv"
REM        run_blurpass process "D:\movie.mkv"
REM        run_blurpass process D:\Videos --recursive
REM        run_blurpass selftest
cd /d "%~dp0"
venv\Scripts\python.exe blurpass.py %*
