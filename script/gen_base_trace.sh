#!/bin/bash
source common.sh

##  Loop over the benches one by one
##  No blank characters should exist in suite names or bench names
suites=$( get_suite_names )
for suite in $suites; do
    benches=$( get_bench_names "$suite" )
    for bench in $benches; do
		##  If the argument $1 is not empty, a specific bench is selected to generate trace
		if [ "$1" == "" -o \( "$1" != "" -a "$1" == "$bench" \) ]; then
        	echo "##  gen_base_trace.sh: Generate base_trace for $suite/$bench  ##"
			cd "$build_dir"

			##  Make sure the dir to store trace exists
			base_trace_dir_bench="$base_trace_dir/$suite/$bench"
			makesure_dir_exists "$base_trace_dir_bench"

            ##  Make sure the dir to store log for this suite exists
            ##  Set the file path to store log info and duration info
            log_dir_base_trace_suite="$log_dir_base_trace/$suite"
            makesure_dir_exists "$log_dir_base_trace_suite"

			##  Generating trace
            ##  Get the time stamp before and after execution
            stamp0=$( get_time_ms )
			./${bench}_base_trace "$base_trace_dir_bench" $( get_bench_args $suite $bench ) | tee "$$log_dir_base_trace_suite/$bench.log"
            stamp1=$( get_time_ms )

            ##  Calculate time duration and write to corresponding file
            duration=$((stamp1 - stamp0))
            echo $duration > "$log_dir_base_trace_suite/$bench.duration"
		fi
    done
done
