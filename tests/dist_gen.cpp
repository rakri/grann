#include "random_dist_gen.h"

int main(int argc, char* argv[]) {
    if(argc != 5) {
        std::cout
        << "Usage: " << argv[0]
        << " [gaussian / cube] [num_vectors] [num_dimensions] [filename] " << std::endl;
    }
    else {
        if(strcmp(argv[1], "gaussian") == 0) grann::random_gaussian(argv[4], atoi(argv[2]), atoi(argv[3]));
        else grann::random_cube(argv[4], atoi(argv[2]), atoi(argv[3]));
    }

    return 0;
}