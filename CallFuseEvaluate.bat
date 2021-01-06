@ECHO OFF 

REM Parse parameters
if "%1" == "" goto End  REM root dir to be processed
if "%2" == "" goto End  REM GPU index
if "%3" == "" goto End  REM max image size
if "%4" == "" goto End  REM NCC win size 

goto MyCommand

:MyCommand
echo do fusion...
colmap stereo_fusion ^
--workspace_path %1/dense ^
--workspace_format COLMAP ^
--input_type geometric ^
--StereoFusion.max_image_size %3 ^
--StereoFusion.max_normal_error 25 ^
--StereoFusion.min_num_pixels 3 ^
--output_path %1/dense/fused_win%4_filled.ply
echo point cloud fusion done 

set TestRoot=%1/dense
set EvalRoot=%1/dslr_scan_eval

echo do ETH3D evaluation...
ETH3DMultiViewEvaluation ^
--reconstruction_ply_path %TestRoot%\fused_win%4_filled.ply ^
--ground_truth_mlp_path %EvalRoot%\scan_alignment.mlp ^
--tolerances 0.01,0.02,0.05,0.1,0.2,0.5
echo evaluation done

goto End

:End
echo End of processing
exit
