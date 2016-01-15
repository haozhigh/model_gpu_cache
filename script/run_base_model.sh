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
        	echo "##  run_base_model.sh: Running base_model for $suite/$bench  ##"

            ##  Get the time stamp before and after execution of all kernels of a bench
            stamp0=$( get_time_ms )

            ##  Make sure the dir to store log for this suite exists
            ##  Set the file path to store log info and duration info
            log_dir_base_model_suite="$log_dir_base_model/$suite"
            makesure_dir_exists "$log_dir_base_model_suite"

            ##  Make log_file empty
            ##  One log file to store all kernel modeling log
            echo "" > "$log_dir_base_model_suite/$bench.log"

            ##  Get all kernel name of this bench
			base_trace_dir_bench="$base_trace_dir/$suite/$bench"
            cd "$base_trace_dir_bench"
            trace_file_names=$( ls *.trc )
            kernel_names=$( strip_extensions "$trace_file_names" )

            ##  cd to the build dir
            cd "$build_dir"

            ##  Loop over all kernels in the same bench
            for kernel_name in $kernel_names; do
                
                ##  Run the model
                ./base_model $bench $suite $kernel_name | tee -a "$log_dir_base_model_suite/$bench.log"
            done

            ##  Calculate time duration and write to corresponding file
            stamp1=$( get_time_ms )
            duration=$((stamp1 - stamp0))
            echo $duration > "$log_dir_base_model_suite/$bench.duration"
		fi
    done
done
