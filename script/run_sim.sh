#!/bin/bash
source common.sh

##  Loop over the benches one by one
##  No blank characters should exist in suite names or bench names
suites=$( get_suite_names )
for suite in $suites; do
    benches=$( get_bench_names "$suite" )
    for bench in $benches; do
		##  If the argument $1 is not empty, a specific bench is selected to run sim
		if [ "$1" == "" -o \( "$1" != "" -a "$1" == "$bench" \) ]; then
        	echo "##  run_sim.sh: Running GPGPU-Sim for $suite/$bench  ##"
			cd "$build_dir"

			##  Make sure the dir to store sim output for this suite exists
            out_sim_dir_suite="$out_sim_dir/$suite"
			makesure_dir_exists "$out_sim_dir_suite"

            ##  Make sure the dir to store log for this suite exists
            log_dir_sim_suite="$log_dir_sim/$suite"
            makesure_dir_exists "$log_dir_sim_suite"

			##  Generating trace
            ##  Get the time stamp before and after execution
            stamp0=$( get_time_ms )
			./${bench}_sim $( get_bench_args $suite $bench )  | tee "$out_sim_dir_suite/${bench}.simlog"
            stamp1=$( get_time_ms )

            ##  Calculate time duration and write to corresponding file
            duration=$((stamp1 - stamp0))
            echo $duration > "$log_dir_sim_suite/${bench}.duration"

            cd "$build_dir"
            rm -f _cuobjdump_*
		fi
    done
done
