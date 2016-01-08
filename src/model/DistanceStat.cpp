



#include <iostream>
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

void DistanceStat::write_to_file(std::string file_path) {
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
}
