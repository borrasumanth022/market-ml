@echo off
cd "C:\Users\borra\OneDrive\Desktop\ML Projects\market_ml"
echo Running Regime Agent...
C:\Users\borra\anaconda3\python.exe src/agents/regime_agent.py
echo Running Signal Generator...
C:\Users\borra\anaconda3\python.exe src/pipeline/12_signal_generator.py
pause
