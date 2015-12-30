


#include <iostream>
#include <vector>
#include <string>

#ifndef MY_WARP_ACCESS
#define MY_WARP_ACCESS


class GlobalAccess {
    public:
    unsigned long long address;
    int width;

    GlobalAccess();
};


#endif
