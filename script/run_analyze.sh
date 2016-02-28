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
        	echo "##  run_analyze.sh: Running analyze for $suite/$bench  ##"

            ##  Get the time stamp before and after analysis of all kernels of a bench
            stamp0=$( get_time_ms )

            ##  Make sure the dir to store analysis result for this suite exists
            out_analyze_dir_bench="$out_analyze_dir/$suite/$bench"
            makesure_dir_exists "$out_analyze_dir_bench"

            ##  Make sure the dir to store log for this suite exists
            ##  Set the file path to store log info and duration info
            log_dir_analyze_suite="$log_dir_analyze/$suite"
            makesure_dir_exists "$log_dir_analyze_suite"

            ##  Make log_file empty
            ##  One log file to store all kernel analysis
            echo "" > "$log_dir_analyze_suite/$bench.log"

            ##  Get all kernel name of this bench
			trace_dir_bench="$trace_dir/$suite/$bench"
            cd "$trace_dir_bench"
            trace_file_names=$( ls *.trc )
            kernel_names=$( strip_extensions "$trace_file_names" )

            ##  cd back to scritp_dir
            cd $script_dir

            ##  Loop over all kernels in the same bench
            for kernel_name in $kernel_names; do

                ##  Run the analysis
                ./analyze.py $suite $bench $kernel_name
            done

            ##  Calculate time duration and write to corresponding file
            stamp1=$( get_time_ms )
            duration=$((stamp1 - stamp0))
            echo $duration > "$log_dir_analyze_suite/$bench.duration"
		fi
    done
done
