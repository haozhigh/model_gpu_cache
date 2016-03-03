#!/bin/bash
source common.sh

nvprof_path="/usr/local/cuda-5.5/bin/nvprof"
nvprof_flags="--metrics l1_cache_global_hit_rate,l2_l1_read_hit_rate"

##  Loop over the benches one by one
##  No blank characters should exist in suite names or bench names
suites=$( get_suite_names )
for suite in $suites; do
    benches=$( get_bench_names "$suite" )
    for bench in $benches; do
		##  If the argument $1 is not empty, a specific bench is selected to run the profiler version
		if [ "$1" == "" -o \( "$1" != "" -a "$1" == "$bench" \) ]; then
        	echo "##  run_profiler.sh: Running profiler for $suite/$bench  ##"
			cd "$build_dir"

			##  Make sure the dir to store profiler output for this suite exists
            out_dir_suite="$out_profiler_dir/$suite"
			makesure_dir_exists "$out_dir_suite"

            ##  Make sure the dir to store log for this suite exists
            log_dir_suite="$log_dir_profiler/$suite"
            makesure_dir_exists "$log_dir_suite"

			##  Generating trace
            ##  Get the time stamp before and after execution
            stamp0=$( get_time_ms )
			$nvprof_path $nvprof_flags ./${bench}_profiler $( get_bench_args $suite $bench ) 2> "$out_dir_suite/${bench}.prof" | tee "$log_dir_suite/${bench}.log"
            stamp1=$( get_time_ms )

            ##  Calculate time duration and write to corresponding file
            duration=$((stamp1 - stamp0))
            echo $duration > "$log_dir_suite/${bench}.duration"
		fi
    done
done
