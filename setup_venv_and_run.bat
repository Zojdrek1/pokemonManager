@echo off
setlocal EnableExtensions
chcp 65001 >NUL
cd /d "%~dp0"

REM --- Create venv if missing ---
if not exist ".venv" (
  echo ▶ Creating virtual environment .venv ...
  py -m venv .venv
)

set "PYEXE=%CD%\.venv\Scripts\python.exe"

echo ▶ Upgrading pip/setuptools/wheel...
"%PYEXE%" -m pip install --upgrade pip setuptools wheel

echo ▶ Installing requirements...
if exist requirements.txt (
  "%PYEXE%" -m pip install -r requirements.txt
) else (
  "%PYEXE%" -m pip install streamlit pandas requests beautifulsoup4 lxml pillow
)

REM --- Enforce the AgGrid version that works with your app ---
echo ▶ Ensuring streamlit-aggrid==0.3.4.post3 ...
"%PYEXE%" -m pip install --force-reinstall --no-deps "streamlit-aggrid==0.3.4.post3"

echo --- Versions in venv ---
"%PYEXE%" -m pip show streamlit | findstr /R "Version"
"%PYEXE%" -m pip show streamlit-aggrid | findstr /R "Version"
echo -------------------------

echo ▶ Starting Pokémon Manager...
"%PYEXE%" -m streamlit run app.py
echo.
pause
