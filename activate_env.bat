@echo off
echo Launching Git Bash with fracnn conda environment...
echo.
echo This will open Git Bash and automatically activate the fracnn environment.
echo.
echo If Git Bash is not installed, please install it from: https://git-scm.com/
echo.
pause

REM Launch Git Bash with the activation script
"C:\Program Files\Git\bin\bash.exe" --login -i -c "source activate_env.sh"
