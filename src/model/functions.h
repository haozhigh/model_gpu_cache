#include <string>

#include "Access.h"
#include "ModelConfig.h"



void string_trim(std::string & str);
bool string_is_int(const std::string & str);
int string_to_int(const std::string & str);


int calculate_cache_set(addr_type line_addr, ModelConfig & model_config);
