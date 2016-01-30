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


int calculate_cache_set(addr_type line_addr, ModelConfig & model_config) {
	int set = 0;
	
	// Default mapping function (no 'hash')
	if (model_config.mapping_type == 0) {
		set = line_addr % model_config.cache_set_size;
	}
	
	// Basic XOR hashing function
	else if (model_config.mapping_type == 1) {
		set = (line_addr % model_config.cache_set_size) ^ ((line_addr/model_config.cache_set_size) % model_config.cache_set_size);
	}
	
	// Fermi's hashing function
	else if (model_config.mapping_type == 2) {
        // Generate groups of bits
        int bits[13];
        for (int i = 0; i < 13; i ++) {
            bits[i] = line_addr % 2;
            line_addr = line_addr >> 1;
        }
	
        //  Calculate set
		unsigned b01234 = bits[0] + bits[1]*2 + bits[2]*4 + bits[3] *8 + bits[4] *16;
		unsigned b678AC = bits[6] + bits[7]*2 + bits[8]*4 + bits[10]*8 + bits[12]*16;
		set = (b01234 ^ b678AC) + bits[5]*32;
	}
    
    //  Maxwell texture cache hashing function
    else if (model_config.mapping_type == 3) {
        set = (line_addr >> 2) % model_config.cache_set_size;
    }
	
	// Return the result modulo the number of sets
	return (set % model_config.cache_set_size);

}
