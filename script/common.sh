##  Set bash shell to display every command executed
##set -x

##  Get current time stamp at milli seconds
function get_time_ms {
    t1=`date +%s`
    t1=$(((t1 % 100000) * 1000))
    t2=10#`date +%N`
    t2=$((t2 / 1000000))
    t=$((t1 + t2))
    echo $t
}

##  Make sure that a dir exists, creat it if not
function makesure_dir_exists {
	if [ ! -d "$1" ]; then
		mkdir -p "$1"
	fi
}

##  Get benchmark suite names
function get_suite_names {
	suites=$( ls "$benchmarks_dir" )
	new_suites=""
	for suite in $suites; do
		if [ "$suite" != "common" ]; then
			if [ "$new_suites" = "" ]; then
				new_suites="$suite"
			else
				new_suites="$new_suites $suite"
			fi
		fi
	done
	echo "$new_suites"
}

##  Get benchmark name of a certain suite
function get_bench_names {
	benches=$( ls "$benchmarks_dir/$1" )
	echo "$benches"
}

##  Get args for a certain bench
function get_bench_args {
	args_file="$benchmarks_dir/$1/$2/args"
	args=""
	if [ -f "$args_file" ]; then
		args=$( cat "$args_file" )
	fi
	echo "$args"
}

## Stripe file extensions for a list of file names
function strip_extensions {
    return_str=""
    for original_name in $1; do
        return_str="$return_str ${original_name%.*}"
    done
    echo $return_str
}

## Get the full dir of the script file which sources common.sh
script_dir="$( cd "$( dirname $0 )" && pwd )"

##  Set the specific dirs
build_dir="$script_dir/../build"
log_dir="$script_dir/../log"
log_dir_trace="$log_dir/trace"
log_dir_model="$log_dir/model"
trace_dir="$script_dir/../output/trace"
benchmarks_dir="$script_dir/../src/benchmarks"

model_dir=$"$script_dir/../src/model"

##  Make sure that the above dirs exist
makesure_dir_exists "$build_dir"
makesure_dir_exists "$log_dir"
makesure_dir_exists "$log_dir_trace"
makesure_dir_exists "$log_dir_model"
makesure_dir_exists "$trace_dir"
