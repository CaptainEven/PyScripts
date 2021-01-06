@echo off 

REM Parse parameters
if "%1"=="" (
    goto End
) else (
    goto MyCommand
)

REM Run command
:MyCommand
echo parse parameters done, run...
for /d %%i in (%1\*) do (
    echo processing %%i...
    REM "C:\ColMapMVSMy\x64\Release\ColMapMVSMy" %%i 0
    echo. %%i processed.
)

goto End
:End
echo End of processing.
exit