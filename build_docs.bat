@echo off
setlocal

REM ==========================================
REM PATH CONFIGURATION
REM ==========================================

set SOURCE_DIR=src\tam
set SPHINX_SOURCE=docs\source
set OUTPUT_DIR=docs\build

echo ==========================================
echo Building Time series Additive Model (TAM) Documentation
echo ==========================================

echo.
echo 1. Cleaning previous builds...
if exist "%OUTPUT_DIR%" rmdir /S /Q "%OUTPUT_DIR%"
if exist "%SPHINX_SOURCE%\api" rmdir /S /Q "%SPHINX_SOURCE%\api"
if not exist "%SPHINX_SOURCE%\_static" mkdir "%SPHINX_SOURCE%\_static"

echo.
echo 2. Scanning source code (sphinx-apidoc)...
sphinx-apidoc -f -q -e -M -o "%SPHINX_SOURCE%\api" "%SOURCE_DIR%" "%SOURCE_DIR%\model" "%SOURCE_DIR%\evaluation"

echo.
echo 3. Building HTML Website...
sphinx-build -q -b html "%SPHINX_SOURCE%" "%OUTPUT_DIR%\html"
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo 4. Building Comprehensive PDF Manual...
REM No need for custom master_doc flags anymore, Sphinx will use the index
sphinx-build -q -b latex "%SPHINX_SOURCE%" "%OUTPUT_DIR%\latex_manual"
if %ERRORLEVEL% NEQ 0 goto :error

REM Use pushd to safely change directory and popd to return
pushd "%OUTPUT_DIR%\latex_manual"
REM Compile the Master PDF
pdflatex -interaction=nonstopmode TAM_documentation.tex
makeindex -s python.ist TAM_documentation.idx
bibtex TAM_documentation.aux
pdflatex -interaction=nonstopmode TAM_documentation.tex
pdflatex -interaction=nonstopmode TAM_documentation.tex
popd

echo.
echo ==========================================
echo SUCCESS: HTML Site and PDF generated successfully!
echo.
echo 🌐 HTML : %OUTPUT_DIR%\html\index.html
echo 📕 PDF  : %OUTPUT_DIR%\latex_manual\TAM_documentation.pdf
echo ==========================================
exit /b 0

:error
echo ==========================================
echo ERROR: Documentation build failed. Check the logs above.
echo ==========================================
exit /b 1