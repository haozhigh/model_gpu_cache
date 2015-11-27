#!/bin/bash

source common.sh

## Get the full dir of this script file
cur_dir="$( cd "$( dirname $0 )" && pwd )"

##  Make sure that ../build directory exists
if [ ! -d "$cur_dir/../build" ]; then
	mkdir "$cur_dir/../build"
fi

##  Clean formally made executables if argument specified
if [ "$1" == "clean" ]; then
    echo "##Removing everything in ../build"
    rm -f "$cur_dir/../build/*_profiler"
    rm -f "$cur_dir/../build/*_sim"
    rm -f "$cur_dir/../build/*_trace"
    rm -f "$cur_dir/../build/cachemodel_*"
    exit 0
fi

##  Do the make one bench by another
##  No blank characters should exist in suite names or bench names
cd "$cur_dir/../src/benchmarks/"
suites=`ls`
for suite in $suites; do
    if [ $suite != common ]; then
        cd $suite
        benches=`ls`
        for bench in $benches; do
            echo "####Start making bench: $suite/$bench"
            cd $bench
            make
            cd ..
            echo
        done
        cd ..
    fi
done

#cd "$cur_dir/../src/model_base"
#echo "####Start making cache model"
#make
#echo
