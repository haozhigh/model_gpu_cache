



#include <random>




class NormalGenerator {
    private:
    int mean;
    int stddev;
    std::random_device rd;
    std::mt19937 * gen;
    std::normal_distribution<float> * d;

    public:
    NormalGenerator(int mean, int stddev);
    NormalGenerator();
    ~NormalGenerator();
    int next_number();
};
