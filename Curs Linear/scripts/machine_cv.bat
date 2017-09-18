@echo off
start /b /d "..\LinRegCurs\Debug" /wait LinRegCurs.exe crossValidation machine_cpu.features model_machine welford_lr ISYT_machine_sse5 10 5
start /b /d "..\LinRegCurs\Debug" /wait LinRegCurs.exe crossValidation machine_cpu.features model_machine welford_lr ISYT_machine_sse10 10 10
start /b /d "..\LinRegCurs\Debug" /wait LinRegCurs.exe crossValidation machine_cpu.features model_machine welford_lr ISYT_machine_sse15 10 15
start /b /d "..\LinRegCurs\Debug" /wait LinRegCurs.exe crossValidation machine_cpu.features model_machine welford_lr ISYT_machine_sse20 10 20