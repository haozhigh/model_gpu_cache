#!/usr/bin/python3



from pandas import Series, DataFrame
from parser import *




def main():
    wide_kernel_names = get_wide_kernel_names_trace()
    miss_frame = DataFrame(index = wide_kernel_names)

    parse_model_out(miss_frame, wide_kernel_names)
    print(miss_frame)



if __name__ == '__main__':
    main()
