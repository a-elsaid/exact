#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

int main() {
    std::vector<int> ints;
    for (int i=1; i<8; i++){
        ints.push_back(i);
    }
    
    for (int j=0; j<10; j++) {
        std::default_random_engine rng;
        std::shuffle(ints.begin(), ints.end(), rng);
        for (auto&& i : ints) {
            std::cout << i << ' ';
        }
        std::cout << std::endl;
    }
}
