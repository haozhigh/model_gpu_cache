



#include "NormalGenerator.h"


NormalGenerator::NormalGenerator(int _mean, int _stddev) {
    mean = _mean;
    stddev = _stddev;

    gen = new std::mt19937(rd());
    d = new std::normal_distribution<float>(0, stddev);
}

NormalGenerator::NormalGenerator() {
    NormalGenerator(0, 0);
}

int NormalGenerator::next_number() {
   return mean + std::abs(std::round(d->operator()((*gen))));
}

NormalGenerator::~NormalGenerator() {
    delete d;
    delete gen;
}
