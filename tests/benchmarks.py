import subprocess
import sys
import os

datasets = ["SIFT1M", "Gaussian", "Cube"]

def vamana():
    k = 10
    R = [50, 75, 100]
    L = [1, 1.5, 2]
    alpha = [1, 1.2, 1.4]

    for r in R:
        for l in L:
            for a in alpha:
                dirpath = f"/mnt/kairav/exp/vamana/{r}_{int(l*r)}_{a}"

                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)
                    os.makedirs(dirpath + "/search")

                with open(dirpath + "/build.txt", "w") as outfile:
                    subprocess.run(["../build/tests/build_vamana", "float", "l2", "/mnt/SIFT1M/base.bin", dirpath + "/index", f"{r}", f"{int(l*r)}", f"{a}", "64"], stdout=outfile)
            
                with open(dirpath + f"/search_{k}.txt", "w") as outfile:
                    args = ["../build/tests/search_vamana", "float", "l2", dirpath + "/index", "64", "/mnt/SIFT1M/query.bin", "/mnt/SIFT1M/groundtruth.bin", f"{k}", dirpath + "/search" + f"/search_{k}"]

                    for i in range(0,20):
                        args.append(f"{r + i * 10}")

                    subprocess.run(args, stdout=outfile)


def hnsw():
    k = 10
    sampling_rate = 0.1
    R = [50, 75, 100]
    L = [1, 1.5, 2]
    num_levels = [3, 4]

    for r in R:
        for l in L:
            for levels in num_levels:
                dirpath = f"/mnt/kairav/exp/hnsw/{r}_{int(l*r)}_{levels}"

                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)
                    os.makedirs(dirpath + "/search")

                with open(dirpath + "/build.txt", "w") as outfile:
                    subprocess.run(["../build/tests/build_hnsw", "float", "l2", "/mnt/SIFT1M/base.bin", dirpath + "/index", f"{r}", f"{int(l*r)}", 1, sampling_rate, levels, "64"], stdout=outfile)
            
                with open(dirpath + f"/search_{k}.txt", "w") as outfile:
                    args = ["../build/tests/search_hnsw", "float", "l2", dirpath + "/index", levels, "64", "/mnt/SIFT1M/query.bin", "/mnt/SIFT1M/groundtruth.bin", f"{k}", dirpath + "/search" + f"/search_{k}"]

                    for i in range(0,20):
                        args.append(f"{r + i * 10}")

                    subprocess.run(args, stdout=outfile)


def ivf():
    num_clusters = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
    training_rate = 0.3
    k = 10

    for clusters in num_clusters:
        dirpath = f"/mnt/kairav/exp/ivf/{clusters}"

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
            os.makedirs(dirpath + "/search")

        with open(dirpath + "/build.txt", "w") as outfile:
            subprocess.run(["../build/tests/build_ivf", "float", "/mnt/SIFT1M/base.bin", dirpath + "/index", clusters, training_rate], stdout=outfile)

        with open(dirpath + f"/search_{k}.txt", "w") as outfile:
            args = ["../build/tests/search_ivf", "float", dirpath + "/index", 64, "/mnt/SIFT1M/query.bin", "/mnt/SIFT1M/groundtruth.bin", f"{k}", dirpath + "/search" + f"/search_{k}"]

            for p in range(clusters / 20, clusters + 1, clusters / 20):
                args.append(f"{p}")

            subprocess.run(args, stdout=outfile)


if __name__ == "__main__":
    # vamana()
    hnsw()
    ivf()