@echo off

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
set command="C:\ETH3DMultiViewEvaluation\ETH3DMultiViewEvaluation"
for /d %%i in (%1\*) do (
    echo processing %%i...
    
    echo evaluate colmap...
    %command% --reconstruction_ply_path %%i\src_fused.ply --ground_truth_mlp_path %%i\dslr_scan_eval\scan_alignment.mlp --tolerances 0.01,0.02,0.05,0.1,0.2,0.5

    echo evaluate SelJBPF...
    %command% --reconstruction_ply_path %%i\fusedSelJBPF.ply --ground_truth_mlp_path %%i\dslr_scan_eval\scan_alignment.mlp --tolerances 0.01,0.02,0.05,0.1,0.2,0.5

    echo evaluate NAF...
    %command% --reconstruction_ply_path %%i\fusedNAF.ply --ground_truth_mlp_path %%i\dslr_scan_eval\scan_alignment.mlp --tolerances 0.01,0.02,0.05,0.1,0.2,0.5

    echo. %%i processed.
)

goto End
:End
echo End of processing.
exit