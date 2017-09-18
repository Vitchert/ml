@echo off

start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation cpu_act.features model_cpu welford_lr cpu_sse5000 1 5000
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation puma32H.features model_puma welford_lr puma_sse5000 1 5000
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation delta_ailerons.features model_delta welford_lr delta_sse5000 1 5000
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation kin8nm.features model_kin8 welford_lr kin8_sse5000 1 5000
start /b /d "..\LinRegCurs\Debug" LinRegCurs.exe crossValidation machine_cpu.features model_machine welford_lr machine_sse5000 1 5000