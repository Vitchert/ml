@echo off

start /b /d "..\LinRegCurs\Debug" /wait LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse 10 10

