@echo off
rem 设置 Python 脚本的路径
set SCRIPT_PATH=.\allmain.py

rem 使用 Python 运行脚本
.\python-3.10.10-embed-amd64\python.exe %SCRIPT_PATH%

rem 暂停以便查看输出（可选）
pause
