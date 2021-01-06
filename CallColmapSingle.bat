@ECHO OFF 

REM Parse parameters
if "%1" == "" goto End  REM root dir to be processed
if "%2" == "" goto End  REM GPU index
if "%3" == "" goto End  REM max image size
if "%4" == "" goto End  REM NCC win size 

goto MyCommand

:MyCommand
echo Parse parameters done, running...

if not exist %1 (
    md %1\dense
    echo %1 not exists, made %1\dense
) 
REM else (
REM     rd /s /q %1\dense
REM     md %1\dense
REM )

colmap image_undistorter ^
--image_path %1/images ^
--input_path %1/dslr_calibration_undistorted ^
--output_path %1/dense ^
--output_type COLMAP ^
--max_image_size %3

colmap patch_match_stereo ^
--workspace_path %1/dense ^
--workspace_format COLMAP ^
--PatchMatchStereo.geom_consistency true ^
--PatchMatchStereo.gpu_index %2 ^
--PatchMatchStereo.ncc_sigma 0.6 ^
--PatchMatchStereo.window_radius %4 ^
--PatchMatchStereo.max_image_size %3 ^
--PatchMatchStereo.filter false

colmap stereo_fusion ^
--workspace_path %1/dense ^
--workspace_format COLMAP ^
--input_type geometric ^
--StereoFusion.max_image_size %3 ^
--StereoFusion.max_normal_error 25 ^
--StereoFusion.min_num_pixels 3 ^
--output_path %1/dense/fused_win%4_filled.ply

goto End

:End
echo End of processing
exit
