@ECHO OFF 

REM Parse parameters
if "%1"=="" (
    goto End
) else (
    goto MyCommand
)

REM Run command
:MyCommand
echo Parse parameters done, running...

for /d %%i in (%1\*) do (
    REM if not "%%i" == "%1\courtyard" if not "%%i" == "%1\delivery_area" (
    REM     echo processing %%i...
    REM     "G:\ColMapMVSMy\x64\Release\ColMapMVSMy" %%i 0
    REM     ren %%i\fusedPoint.ply fusedNAF.ply
    REM     echo. %%i processed
    REM )

    echo processing %%i...
    "G:\ColMapMVSMy\x64\Release\ColMapMVSMy" %%i 0
    ren %%i\fusedPoint.ply fused.ply
    echo %%i processed
)

goto End
:End
echo End of processing
exit
