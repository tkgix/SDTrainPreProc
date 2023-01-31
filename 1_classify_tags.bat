set DIR_IN=C:\


set TAGS=sketch,monochrome,no_humans
set DIR_OUT_TAG=%DIR_IN%\_trash_tag

set THRESH=0.1

python.exe classify_files_by_tag.py ^
--dir_in "%DIR_IN%" ^
--thresh %THRESH% ^
--tags %TAGS% ^
--dir_out "%DIR_OUT_TAG%"

pause

