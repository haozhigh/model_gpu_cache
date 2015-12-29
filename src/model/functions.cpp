#include <cctype>
#include "functions.h"

void string_trim(std::string & str) {
    int l, r;

    //  l refers to the location of leftest non white char
    l = 0;
    while (l < str.size() && std::isspace(str[l]))
        l ++;
    //  r refers to the location of the rightest non white char
    r = str.size() - 1;
    while (l >= 0 && std::isspace(str[r]))
        r --;

    //  if str contains no non white space
    if (l >= str.size() || r < 0 || l > r) {
        str = "";
        return;
    }

    //  if nothing needs to be trimed
    if (l == 0 && (r == str.size() - 1))
        return;

    //  trim str
    str = str.substr(l, r - l + 1);
    return;
}

bool string_is_int(const std::string &str) {
    int i;
    std::string str_cpy;

    str_cpy = str;
    string_trim(str_cpy);

    if (str_cpy[0] == '-') {
        i = 1;
        if (str_cpy.size() == 1)
            return false;
    }
    else {
        i = 0;
        if (str_cpy.size() == 0)
            return false;
    }
        
    for (; i < str_cpy.size(); i++) {
        if (str_cpy[i] < '0' || str_cpy[i] > '9')
            return false;
    }

    return true;
}

int string_to_int(const std::string & str) {
    int i;
    std::string str_cpy;
    int sum;

    str_cpy = str;
    string_trim(str_cpy);

    if (str_cpy[0] == '-') {
        i = 1;
    }
    else {
        i = 0;
    }

    sum = 0;
    for (; i < str_cpy.size(); i++) {
        int n;

        n = str_cpy[i] - '0';
        sum = sum * 10 + n;
    }

    if (str_cpy[0] == '-') {
        return -sum;
    }
    else {
        return sum;
    }
}
