:: 자료 위치
set DIR_IN=G:\AI_TrainTemp\_SDTrainPreProc\_run

:: 학습 해상도. 512, 768, 1024 등등. (64단위).
set RES=1024
:: 사이즈를 맞출 때 확장한다면 true, 자른다면 false
set EXPAND=true


python adjust_size.py ^
--res %RES% ^
--dir_in "%DIR_IN%" ^
--expand %EXPAND%

pause