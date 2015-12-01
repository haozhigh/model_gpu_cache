#!/bin/bash
source common.sh

##  Loop over the benches one by one
##  No blank characters should exist in suite names or bench names
suites=$( get_suite_names )
for suite in $suites; do
    benches=$( get_bench_names "$suite" )
    for bench in $benches; do
		##  If the argument $1 is not empty, a specific bench is selected to generate trace
		if [ $1 == "" -o \( $1 != "" -a $1 == $bench \) ]; then
        	echo "##  Generate trace for $suite/$bench  ##"
			cd "$build_dir"

			##  Make sure trace dir for this bench exists
			trace_store_dir_bench="$trace_store_dir/$suite/$bench"
			makesure_dir_exists "$trace_store_dir_bench"

			##  Generating trace
			./${bench}_trace "$trace_store_dir_bench" $( get_bench_args $suite $bench )
		fi
    done
done
