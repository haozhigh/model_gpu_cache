#!/bin/bash
source common.sh

##  Clean formally made executables if argument specified
make_flag=""
bench_specified=$1
if [ "$1" = "clean" ]; then
    make_flag="clean"
    bench_specified=$2
fi

##  Do the make one bench by another
##  No blank characters should exist in suite names or bench names
suites=$( get_suite_names )
for suite in $suites; do
    benches=$( get_bench_names "$suite" )
    for bench in $benches; do
		##  If the argument $1 is not empty, a specific bench is selected to make
		if [ "$bench_specified" = "" -o \( "$bench_specified" != "" -a "$bench_specified" = "$bench" \) ]; then
        	echo "##  Start making $suite/$bench  ##"
			cd "$benchmarks_dir/$suite/$bench"
        	make $make_flag
        	echo
		fi
    done
done

##  Make the model
if [ "$bench_specified" = "model" ]; then
    cd "$model_dir"
    make $make_flag
fi

##  Make the base model
if [ "$bench_specified" = "base_model" ]; then
    cd "$base_model_dir"
    make $make_flag
fi
