#!/usr/bin/python3

import re

def main():
    file_handle = open('cache1_output.tmp', 'r')
    output = file_handle.read()
    file_handle.close()

    pattern = re.compile(r'^\s*\d+\s*l1_cache_global_hit_rate\s*L1 Global Hit Rate\s*(\S+)%\s*(\S+)%\s*(\S+)%\s*$', re.MULTILINE)
    match = pattern.search(output)

    if match == None:
        print(0, end = '')
        return -1
    else:
        hit_rate = float(match.group(1))
        print('%.2f' % (100 - hit_rate))
        return 0

if __name__ == '__main__':
    main()
