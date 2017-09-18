@echo off

start /b /d "..\LinRegCurs\Debug" /wait LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse 10 10



