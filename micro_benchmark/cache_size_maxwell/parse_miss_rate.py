#!/usr/bin/python3

import re
import sys

def main():

    in_str = ""
    for line in sys.stdin:
        in_str = in_str + line
    
    pattern = re.compile(r'tex_cache_hit_rate\s+Texture Cache Hit Rate\s+(\S+)%\s+(\S+)%\s+(\S+)%')
    match = pattern.search(in_str)

    if match == None:
        print(0, end = '')
        return -1
    else:
        hit_rate = float(match.group(3))
        print('%.2f' % (100 - hit_rate))
        return 0

if __name__ == '__main__':
    main()
