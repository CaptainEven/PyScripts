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

"G:\ColMapMVSMy\x64\Release\ColMapMVSMy" %1 0
ren %1\fusedPoint.ply fused.ply
echo %1 processed

goto End
:End
echo End of processing.
exit
