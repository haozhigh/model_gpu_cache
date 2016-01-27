



#include <iostream>
#include <iomanip>
#include <fstream>

#include "DistanceStat.h"

DistanceStat::DistanceStat() {
}

void DistanceStat::increase(int pc, int distance) {
    //  If stat for this pc is empty
    if (stat.find(pc) == stat.end()) {
        stat[pc][distance] = 1;
        return;
    }

    //  If stat for this pc is not empty, but stat for the distance is empty
    std::map<int, int> & stat_sub = stat[pc];
    if (stat_sub.find(distance) == stat_sub.end()) {
        stat[pc][distance] = 1;
        return;
    }

    //  If stat for this pc and distance already exists
    stat[pc][distance] = stat[pc][distance] + 1;
    return;
}

void DistanceStat::merge(DistanceStat & stat2) {
    std::map<int, std::map<int, int>> & map1 = this->stat;
    std::map<int, std::map<int, int>> & map2 = stat2.stat;


    std::map<int, std::map<int, int>>::iterator it1, it3;
    std::map<int , int>::iterator it2, it4;

    int pc;
    int distance;
    int count;
    for (it1 = map2.begin(); it1 != map2.end(); it1++) {
        pc = it1->first;

        for (it2 = it1->second.begin(); it2 != it1->second.end(); it2++) {
            distance = it2->first;
            count = it2->second;

            //  Check if (pc, distance) pare already exists in map1
            bool pair_exist = false;
            it3 = map1.find(pc);
            if (it3 != map1.end()) {
                it4 = it3->second.find(distance);
                if (it4 != it3->second.end()) {
                    pair_exist = true;
                }
            }

            //  update map1
            if (! pair_exist)
                map1[pc][distance] = 0;
            map1[pc][distance] += count;
        }
    }
}

void DistanceStat::calculate_miss_rate(const ModelConfig & model_config) {
    int hit_count, compulsory_miss_count, uncompulsory_miss_count;
    int total_count;

    std::map<int, std::map<int, int>>::iterator it1;
    std::map<int, int>::iterator it2;

    hit_count = 0;
    compulsory_miss_count = 0;
    uncompulsory_miss_count = 0;
    for (it1 = stat.begin(); it1 != stat.end(); it1 ++) {
        for (it2 = it1->second.begin(); it2 != it1->second.end(); it2 ++) {
            int distance;
            int count;

            distance = it2->first;
            count = it2->second;
            if (distance < 0) {
                compulsory_miss_count += count;
            }
            else {
                if (distance < model_config.cache_way_size) {
                    hit_count += count;
                }
                else {
                    uncompulsory_miss_count += count;
                }
            }
        }
    }

    total_count = hit_count + compulsory_miss_count + uncompulsory_miss_count;
    compulsory_miss_rate = (double)1.0 * compulsory_miss_count / total_count;
    uncompulsory_miss_rate = (double)1.0 * uncompulsory_miss_count / total_count;
}

void DistanceStat::write_miss_rate_to_file(std::string file_path) {
    std::ofstream out_stream;
    std::string miss_rate_file_path;

    //  Get miss rate output file path
    //miss_rate_file_path = file_path.substr(0, file_path.size() - 9) + ".miss_rate";
    miss_rate_file_path = file_path + ".miss_rate";

    //  Open the file for writing
    out_stream.open(miss_rate_file_path, std::ofstream::out);
    if (! out_stream.is_open()) {
        std::cout << "####  DistanceStat::write_miss_rate_to_file: Failed to open file '" << miss_rate_file_path << "' for writing" << std::endl;
        return;
    }

    //  Write to file
    out_stream << std::fixed << std::setprecision(4) << compulsory_miss_rate << " " << uncompulsory_miss_rate << " " << compulsory_miss_rate + uncompulsory_miss_rate;

    //  Close file
    out_stream.close();
}

void DistanceStat::write_to_file(std::string file_path, const ModelConfig & model_config) {
    std::ofstream out_stream;

    //  Open the file for writing
    out_stream.open(file_path, std::ofstream::out);
    if (! out_stream.is_open()) {
        std::cout << "####  DistanceStat::write_to_file: Failed to open file '" << file_path << "' for writing" << std::endl;
        return;
    }

    //  Write stat to file
    std::map<int, std::map<int, int>>::iterator it1;
    std::map<int, int>::iterator it2;

    for (it1 = stat.begin(); it1 != stat.end(); it1 ++) {
        for (it2 = it1->second.begin(); it2 != it1->second.end(); it2 ++) {
            out_stream << it1->first << " " << it2->first << " " << it2->second << std::endl;
        }
    }

    //  Close file
    out_stream.close();


    //  Compute miss rate
    this->calculate_miss_rate(model_config);

    //  Output miss rate
    this->write_miss_rate_to_file(file_path);
}
