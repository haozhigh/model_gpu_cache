#nvprof --events l1_global_load_hit,l1_global_load_miss,l1_local_load_hit,l1_local_load_miss --metrics l1_cache_global_hit_rate ./cache
#nvprof --metrics l1_cache_global_hit_rate ./cache

ARRAY_SIZE_START=1500     #in number of ints
ARRAY_SIZE_END=5000
STRIDE_SET=(10 23)

echo "stride,array_size,l1_miss_rate" > cache1.csv

for stride in "${STRIDE_SET[@]}"; do
    array_size=$ARRAY_SIZE_START
    while [ $array_size -le $ARRAY_SIZE_END ]; do
        ##  Run the profiler, store the result in temp file cache1_output.tmp
        nvprof --metrics l1_cache_global_hit_rate ./cache1 $array_size $stride 2> cache1_output.tmp

        ##  Call the python script to parse nvprof output
        miss_rate=`./parse_cache1_output.py`

        ##  Echo the result and store it to cache1.csv file
        echo $stride,$array_size,$miss_rate
        echo $stride,$array_size,$miss_rate >> cache1.csv

        ##  Increase array size
        array_size=$[$array_size + 1]
    done
done

##  Delete the temp file
rm cache1_output.tmp
