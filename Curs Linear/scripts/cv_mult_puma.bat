@echo off

start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse20 10 20
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse19 10 19
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse18 10 18
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse17 10 17
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse16 10 16
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse15 10 15
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse14 10 14
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse13 10 13
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse12 10 12
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse11 10 11
goto end
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse10 10 10
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse9 10 9
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse8 10 8
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse7 10 7
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse6 10 6
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse5 10 5
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse4 10 4
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse3 10 3
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse2 10 2
:end
