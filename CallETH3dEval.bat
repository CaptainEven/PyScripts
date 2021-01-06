@ECHO OFF

REM set TestRoot="E:\ETH3D\multi_view_training_dslr_undistorted\office"
set TestRoot="C:\office\dense"
set EvalRoot="C:\office\dslr_scan_eval"

REM echo processing colmap
REM ETH3DMultiViewEvaluation ^
REM --reconstruction_ply_path %TestRoot%\fused_win5.ply ^
REM --ground_truth_mlp_path %EvalRoot%\scan_alignment.mlp ^
REM --tolerances 0.01,0.02,0.05,0.1,0.2,0.5
REM echo colmap processed

echo processing colmap filled
ETH3DMultiViewEvaluation ^
--reconstruction_ply_path %TestRoot%\fused_win5_filled.ply ^
--ground_truth_mlp_path %EvalRoot%\scan_alignment.mlp ^
--tolerances 0.01,0.02,0.05,0.1,0.2,0.5
echo colmap filled processed

REM echo processing enhance colmap
REM F:\ETH3DMultiViewEvaluation\ETH3DMultiViewEvaluation --reconstruction_ply_path %TestRoot%\sparse_office_enh.ply --ground_truth_mlp_path %EvalRoot%\scan_alignment.mlp --tolerances 0.01,0.02,0.05,0.1,0.2,0.5
REM echo enhance colmap processed
