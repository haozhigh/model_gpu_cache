#!/bin/bash
source common.sh

##  Loop over the benches one by one
##  No blank characters should exist in suite names or bench names
suites=$( get_suite_names )
for suite in $suites; do
    benches=$( get_bench_names "$suite" )
    for bench in $benches; do
		##  If the argument $1 is not empty, a specific bench is selected to generate code
		if [ "$1" == "" -o \( "$1" != "" -a "$1" == "$bench" \) ]; then
        	echo "##  Generate code for $suite/$bench  ##"
			cd "$build_dir"

			##  Make sure the dir to store code exists
			code_dir_bench="$code_dir/$suite/$bench"
			makesure_dir_exists "$code_dir_bench"

            ##  Make sure the dir to store log for this suite exists
            ##  Set the file path to store log info and duration info
            log_dir_code_suite="$log_dir_code/$suite"
            makesure_dir_exists "$log_dir_code_suite"
            log_file="$log_dir_code_suite/$bench.log"
            duration_file="$log_dir_code_suite/$bench.duration"

			##  Generating code
            ##  Get the time stamp before and after execution
            stamp0=$( get_time_ms )
			./${bench}_code "$code_dir_bench" $( get_bench_args $suite $bench ) | tee "$log_file"
            stamp1=$( get_time_ms )

            ##  Calculate time duration and write to corresponding file
            duration=$((stamp1 - stamp0))
            echo $duration > "$duration_file"
		fi
    done
done
