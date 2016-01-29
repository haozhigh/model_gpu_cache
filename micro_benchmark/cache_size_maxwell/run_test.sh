#nvprof --events l1_global_load_hit,l1_global_load_miss,l1_local_load_hit,l1_local_load_miss --metrics l1_cache_global_hit_rate ./cache
#nvprof --metrics l1_cache_global_hit_rate ./cache

ARRAY_SIZE_START=2944  #in number of ints
ARRAY_SIZE_END=3584
STRIDE_SET=(4)

echo "stride,array_size,l1_miss_rate" > result.csv

for stride in "${STRIDE_SET[@]}"; do
    array_size=$ARRAY_SIZE_START
    while [ $array_size -le $ARRAY_SIZE_END ]; do

        ##  Do
        miss_rate=`nvprof --metrics tex_cache_hit_rate ./main $array_size $stride 2>&1 >/dev/null | ./parse_miss_rate.py`

        ##  Echo the result and store it to cache1.csv file
        echo $stride,$array_size,$miss_rate
        echo $stride,$array_size,$miss_rate >> result.csv

        ##  Increase array size
        array_size=$[$array_size + 1]
    done
done
