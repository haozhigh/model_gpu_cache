#!/bin/bash

source common.sh

model_config_file="$build_dir/model_config/opt_break_trace_on_fermi_16k.config"


##  Loop over the benches one by one
##  No blank characters should exist in suite names or bench names
suites=$( get_suite_names )
for suite in $suites; do
    benches=$( get_bench_names "$suite" )
    for bench in $benches; do
		##  If the argument $1 is not empty, a specific bench is selected to generate trace
		if [ "$1" == "" -o \( "$1" != "" -a "$1" == "$bench" \) ]; then
        	echo "##  opt_break_run_trace_on.sh: Running model for $suite/$bench  ##"

            ##  Get the time stamp before and after execution of all kernels of a bench
            stamp0=$( get_time_ms )

            ##  Make sure the dir to store model output for this suite exists
            out_dir_bench="$out_opt_break_dir/trace_on/$suite/$bench"
            makesure_dir_exists "$out_dir_bench"

            ##  Make sure the dir to store log for this suite exists
            log_dir_suite="$log_dir_opt_break/trace_on/$suite"
            makesure_dir_exists "$log_dir_suite"

            ##  Make log_file empty
            ##  One log file to store all kernel modeling log
            echo "" > "$log_dir_suite/$bench.log"

            ##  Get all kernel name of this bench
            ##  Input and output files are in the same directory
			trace_dir_bench="$trace_dir/$suite/$bench"
            cd "$trace_dir_bench"
            trace_file_names=$( ls *.trc )
            kernel_names=$( strip_extensions "$trace_file_names" )

            ##  cd to the build dir
            cd "$build_dir"

            ##  Loop over all kernels in the same bench
            for kernel_name in $kernel_names; do
                
                ##  Run the model
                ./model "$trace_dir_bench/${kernel_name}.trc" "$out_dir_bench/${kernel_name}.distance" "$model_config_file" | tee -a "$log_dir_suite/$bench.log"
            done

            ##  Calculate time duration and write to corresponding file
            stamp1=$( get_time_ms )
            duration=$((stamp1 - stamp0))
            echo $duration > "$log_dir_suite/$bench.duration"
		fi
    done
done
