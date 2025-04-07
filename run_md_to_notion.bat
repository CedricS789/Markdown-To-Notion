@echo off
cls
REM Interactive batch file to run Markdown to Notion script
REM Asks whether to use a file path or direct text input.
REM Corrected version: Omits --page_id flag entirely when prompt is blank.

REM --- !!! IMPORTANT: ADJUST THIS PATH IF NEEDED !!! ---
set PROJECT_DIR=.\
REM ---

set PYTHON_EXEC=%PROJECT_DIR%\.venv\Scripts\python.exe
set SCRIPT_PATH=%PROJECT_DIR%\md_to_notion.py
set ENV_FILE=%PROJECT_DIR%\.env

REM --- Sanity Checks ---
if not exist "%PYTHON_EXEC%" (
    echo ERROR: Python executable not found at "%PYTHON_EXEC%"
    echo Please check the PROJECT_DIR variable and ensure the virtual environment exists.
    pause
    exit /b 1
)
if not exist "%SCRIPT_PATH%" (
    echo ERROR: Python script not found at "%SCRIPT_PATH%"
    echo Please check the PROJECT_DIR variable.
    pause
    exit /b 1
)
 if not exist "%ENV_FILE%" (
    echo WARNING: .env file not found at "%ENV_FILE%"
    echo Default Notion Page ID may not be available if NOTION_PAGE_ID is not in .env.
    echo Ensure NOTION_API_KEY is set in .env or globally.
)
echo.
echo --- Markdown to Notion Uploader (Clickable Version) ---
echo.

REM --- Choose Input Mode ---
:ChooseMode
echo Please choose the input method:
echo   [F] Provide a File Path
echo   [T] Paste/Type Direct Text
echo.
set /p INPUT_MODE="Enter F or T: "

REM Validate Mode (case-insensitive check) and branch
if /i "%INPUT_MODE%"=="F" goto :GetFileInput
if /i "%INPUT_MODE%"=="T" goto :GetTextInput
echo Invalid choice. Please enter only F or T.
echo.
goto ChooseMode

REM --- Get File Input ---
:GetFileInput
    set PYTHON_FLAG=--file
    set "INPUT_DATA="
    echo.
    :PromptFile
    echo --- Enter File Path ---
    set /p INPUT_DATA="Full path to Markdown file: "
    if "%INPUT_DATA%"=="" (
        echo ERROR: File path cannot be empty. Please try again.
        echo.
        goto PromptFile
    )
    if not exist "%INPUT_DATA%" (
        echo ERROR: File not found at "%INPUT_DATA%"
        echo Please check the path and try again.
        echo.
        goto PromptFile
     )
    echo File selected: "%INPUT_DATA%"
    goto GetPageID

REM --- Get Text Input ---
:GetTextInput
    set PYTHON_FLAG=--text
    set "INPUT_DATA="
    echo.
    echo --- Enter Markdown Text ---
    echo Paste or type your Markdown text below and press Enter.
    echo (Note: Pasting multi-line text directly into cmd can sometimes be unreliable.)
    :PromptText
    set /p INPUT_DATA="Markdown Text: "
    if "%INPUT_DATA%"=="" (
        echo ERROR: Input text cannot be empty. Please try again.
        echo.
        goto PromptText
    )
    echo Text input provided.
    goto GetPageID

REM --- Get Notion Page ID ---
:GetPageID
    echo.
    set "NOTION_PAGE_ID="
    echo --- Enter Notion Page ID ---
    echo Enter the Notion Page ID to upload to.
    echo (Leave blank and press Enter to use the default ID from your .env file)
    set /p NOTION_PAGE_ID="Notion Page ID (Optional): "
    echo.

REM --- Run the Python Script ---
echo Processing input...

REM === CORRECTED EXECUTION LOGIC ===
REM Decide which command structure to use based on NOTION_PAGE_ID being empty or not

if not "%NOTION_PAGE_ID%"=="" (
    REM User PROVIDED a Page ID
    echo Target: Specific Notion Page ID ("%NOTION_PAGE_ID%")
    echo Running Python script with specified Page ID...
    echo Command: "%PYTHON_EXEC%" "%SCRIPT_PATH%" %PYTHON_FLAG% "%INPUT_DATA%" --page_id "%NOTION_PAGE_ID%"
    echo.
    "%PYTHON_EXEC%" "%SCRIPT_PATH%" %PYTHON_FLAG% "%INPUT_DATA%" --page_id "%NOTION_PAGE_ID%"
    REM Add --debug like this if needed:
    REM "%PYTHON_EXEC%" "%SCRIPT_PATH%" %PYTHON_FLAG% "%INPUT_DATA%" --page_id "%NOTION_PAGE_ID%" --debug
) else (
    REM User left Page ID BLANK - use default from .env
    echo Target: Default Notion Page ID (from .env)
    echo Running Python script using default Page ID...
    REM Run *WITHOUT* the --page_id argument AT ALL
    echo Command: "%PYTHON_EXEC%" "%SCRIPT_PATH%" %PYTHON_FLAG% "%INPUT_DATA%"
    echo.
    "%PYTHON_EXEC%" "%SCRIPT_PATH%" %PYTHON_FLAG% "%INPUT_DATA%"
    REM Add --debug like this if needed:
    REM "%PYTHON_EXEC%" "%SCRIPT_PATH%" %PYTHON_FLAG% "%INPUT_DATA%" --debug
)
REM === END OF CORRECTED LOGIC ===


REM Check the exit code from the Python script
if %ERRORLEVEL% neq 0 (
  echo WARNING: Python script finished with error code %ERRORLEVEL%. Check output above.
) else (
  echo Python script finished successfully.
)
echo.

echo --- Batch File Finished ---
echo Press any key to close this window.
echo.
pause >nul
