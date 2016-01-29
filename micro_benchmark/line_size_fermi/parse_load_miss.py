#!/usr/bin/python3

import re
import sys

def main():

    in_str = ""
    for line in sys.stdin:
        in_str = in_str + line

    pattern = re.compile(r'\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]')
    match = pattern.search(in_str)

    if match == None:
        print(0, end = '')
        return -1
    else:
        miss_count = int(match.group(1))
        print(miss_count)
        return 0

if __name__ == '__main__':
    main()
