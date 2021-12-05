#include "utils.h"

int main(int argc, char* argv[]) {
    if(argc != 3) {
        std::cout
        << "Usage: " << argv[0]
        << " [gaussian / cube] [filename] " << std::endl;
    }
    else {
        float* data;
        _u64 npts, ndim;

        if(strcmp(argv[1], "gaussian") == 0) grann::load_bin<float>(argv[2], data, npts, ndim);
        else grann::load_bin<float>(argv[2], data, npts, ndim);
    
        for(int i = 0; i < npts; i++) {
            for(int j = 0; j < ndim; j++) {
                grann::cout << data[i * ndim + j] << " ";
            }
            grann::cout << std::endl;
        }
    }

    return 0;
}