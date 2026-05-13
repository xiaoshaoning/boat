@echo off
REM build.bat - Build Nougat-LaTeX OCR example from VS Developer Command Prompt
REM Run this from a Visual Studio 2026 Developer Command Prompt (x64):
REM   "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
REM   cd D:\codes\boat
REM   build.bat

setlocal enabledelayedexpansion

REM Source directories
set BOAT_DIR=D:\codes\boat
set LATEX_DIR=%BOAT_DIR%\examples\latex
set COMMON_DIR=%BOAT_DIR%\examples\common
set BUILD_DIR=%BOAT_DIR%\build

REM Find CUDA
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1

REM Compiler flags
set CFLAGS=/nologo /O2 /MD /std:c11 /W3 /arch:AVX2 /openmp
set CFLAGS=%CFLAGS% /DBOAT_STATIC_BUILD /DBOAT_WITH_CUDA /DBOAT_WITH_SIMD=1 /DBOAT_WITH_OPENMP=1 /DBOAT_USE_OPENBLAS /DBOAT_DEBUG=0
set CFLAGS=%CFLAGS% /I%BOAT_DIR%\include /I%BOAT_DIR%\examples\common
set CFLAGS=%CFLAGS% /I"%CUDA_PATH%\include"

REM OpenBLAS
set CFLAGS=%CFLAGS% /ID:\github\OpenBLAS\build_msvc\install\include
set LFLAGS=%LFLAGS% /LIBPATH:D:\github\OpenBLAS\build_msvc\install\lib openblas.lib

REM CUDA NVCC flags (for .cu files)
set NVCCFLAGS=-allow-unsupported-compiler -D_WINDOWS -Xcompiler="/W3 /GR /EHsc"
set NVCCFLAGS=%NVCCFLAGS% -I%BOAT_DIR%/include -I%BOAT_DIR%/examples/common
set NVCCFLAGS=%NVCCFLAGS% -gencode arch=compute_100,code=sm_100

REM Collect boat sources (CPU)
set BOAT_SRCS=
for %%f in (
    tensor.c memory.c error.c version.c packed.c quantize.c prune.c
    arithmetic.c linear.c simd_kernels.c sgemm.c
    autodiff.c sampling.c
    node.c edge.c graph.c
    attention.c batchnorm.c conv.c dense.c flatten.c gru.c lstm.c norm.c pool.c prelu.c relu.c softmax.c embedding.c
    transformer_decoder.c swin.c
    bpe.c
    cross_entropy.c huber.c loss_common.c mse.c
    graph_model.c model.c sequential.c
    adagrad.c adam.c optimizer_common.c rmsprop.c
    cosine_annealing.c lambda_lr.c scheduler_common.c step_lr.c
    huggingface.c
) do set BOAT_SRCS=!BOAT_SRCS! %BOAT_DIR%\src\layers\%%f

echo Type: %BOAT_SRCS%
echo.
echo This script is a template - please run cmake from VS Developer Command Prompt instead.
echo.
echo Recommended build steps:
echo   1. Open "Visual Studio 2026 Developer Command Prompt"
echo   2. cd D:\codes\boat
echo   3. mkdir build_vs 2^>nul
echo   4. cd build_vs
echo   5. cmake .. -DBOAT_WITH_CUDA=ON -DBOAT_WITH_EXAMPLES=ON -T "cuda=13.1"
echo   6. cmake --build . --target latex_ocr --config Release
echo.
echo NOTE: CUDA 13.1 may not support VS 2026 toolset (v180).
echo If cmake fails, try with Ninja generator:
echo   5. cmake -G Ninja .. -DBOAT_WITH_CUDA=ON -DBOAT_WITH_EXAMPLES=ON ^
echo         -DCMAKE_CUDA_FLAGS_INIT="-allow-unsupported-compiler" ^
echo         -DCMAKE_CUDA_ARCHITECTURES="100"
echo   6. ninja latex_ocr
echo.
pause
