@echo off

start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse100 10 100
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse90 10 90
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse80 10 80
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse70 10 70
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse60 10 60
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse50 10 50
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse40 10 40
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse30 10 30
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse20 10 20
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse10 10 10
goto end
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse10 10 10
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse9 10 9
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse8 10 8
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse7 10 7
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse6 10 6
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse5 10 5
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse4 10 4
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse3 10 3
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse2 10 2
:end

