#!/bin/bash
source common.sh

##  Clean formally made executables if argument specified
if [ "$1" = "clean" ]; then
    echo "##  Removing pre built executables  ##"
	cd "$build_dir"
	echo "##  Cleaning *_profiler  ##"
    rm -f *_profiler
	echo "##  Cleaning *_sim  ##"
    rm -f *_sim
	echo "##  Cleaning *_trace  ##"
    rm -f *_trace
	echo "##  Cleaning cachemodel_*  ##"
    rm -f cachemodel_*
    exit 0
fi

##  Do the make one bench by another
##  No blank characters should exist in suite names or bench names
suites=$( get_suite_names )
for suite in $suites; do
    benches=$( get_bench_names "$suite" )
    for bench in $benches; do
		##  If the argument $1 is not empty, a specific bench is selected to make
		if [ "$1" = "" -o \( "$1" != "" -a "$1" = "$bench" \) ]; then
        	echo "##  Start making $suite/$bench  ##"
			cd "$benchmarks_dir/$suite/$bench"
        	make
        	echo
		fi
    done
done

#cd "$script_dir/../src/model_base"
#echo "####Start making cache model"
#make
#echo
