
##  Do
nvprof --aggregate-mode off --events l1_global_load_miss,l1_global_load_hit ./main 2>&1 | tee result.txt
