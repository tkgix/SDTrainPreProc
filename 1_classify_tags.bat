set DIR_IN=G:\AI_TrainTemp\_SDTrainPreProc\_run


set TAGS=sketch,monochrome,no_humans
set DIR_OUT_TAG=%DIR_IN%\_trash_tag
:: 보통 0.3~ 0.35 로 태깅하나, 저퀄 자료를 좀 더 엄격하게 거르기 위해 0.1
set THRESH=0.1

python.exe classify_files_by_tag.py ^
--dir_in "%DIR_IN%" ^
--thresh %THRESH% ^
--tags %TAGS% ^
--dir_out "%DIR_OUT_TAG%"

pause

