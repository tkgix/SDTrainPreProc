set DIR_IN=C:\

set RES=768
set EXPAND=false

python adjust_size.py ^
--res %RES% ^
--dir_in "%DIR_IN%" ^
--expand %EXPAND%

pause