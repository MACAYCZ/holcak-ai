@echo off

REM Be aware you need to compile the CMakeLists.txt to the same build directory as provided!
cmake --build %1 --target ALL_BUILD --config Release -- /nologo /verbosity:minimal /maxcpucount
