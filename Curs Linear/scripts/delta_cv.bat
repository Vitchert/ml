@echo off

start /b /d "..\LinRegCurs\Debug" /wait LinRegCurs.exe crossValidation delta_ailerons.features model_delta welford_lr delta_sse 10 10




