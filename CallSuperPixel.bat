@echo off
SETLOCAL ENABLEDELAYEDEXPANSION 

REM Parse parameters
if "%1" == "" goto End  REM root dir to be processed
if "%2" == "" goto End  REM number of superpixels
if "%2" == "" goto End  REM superpixel mode

goto MyCommand

:MyCommand
echo Parse parameters done, running...

for %%i in (%1\*.JPG) do (
    "G:\TestCpp\x64\Release\TestGuideFilter" %%i %2 %3
    echo %%i processed
)

goto End

:End
echo End of processing
exit