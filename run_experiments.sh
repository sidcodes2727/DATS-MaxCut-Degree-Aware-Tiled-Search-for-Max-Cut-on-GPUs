#!/bin/bash
# runs all experiments for the paper and writes CSV files in results/
#   ./run_experiments.sh cpu    -> CPU experiments (we ran these on our laptop)
#   ./run_experiments.sh gpu    -> GPU experiments (run on Colab / lab GPU machine)
#   ./run_experiments.sh extra  -> extra CPU experiments (thread scaling, graph size, tile size, convergence)
# after both are done:  python3 paper/make_paper.py

MODE=${1:-cpu}
mkdir -p results data
cd "$(dirname "$0")"

# download datasets if they are not there
for g in G1 G14 G22 G43 G48 G55; do
    [ -f data/$g.txt ] || curl -sfL -o data/$g.txt https://web.stanford.edu/~yyye/yyye/Gset/$g
done
for f in wiki-Vote email-Enron ca-AstroPh soc-Epinions1; do
    [ -f data/$f.txt ] || (curl -sfL -o data/$f.txt.gz https://snap.stanford.edu/data/$f.txt.gz && gunzip -f data/$f.txt.gz)
done

build_cpu() { # $1 = output name, $2 = extra flags
    if [ "$(uname)" == "Darwin" ]; then
        OMP=$(brew --prefix libomp)
        clang++ -std=c++17 -O3 $2 -Xpreprocessor -fopenmp -I$OMP/include -L$OMP/lib -lomp maxcut_research.cpp -o $1 || exit 1
    else
        g++ -O3 $2 -fopenmp maxcut_research.cpp -o $1 || exit 1
    fi
}

if [ "$MODE" == "extra" ]; then
    # extra CPU experiments used in the longer version of the paper
    build_cpu dats ""
    rm -f results/cpu_threads.csv results/cpu_scaling.csv results/cpu_tilesize.csv results/convergence.csv
    # 1. OpenMP scaling (10 searches, different number of threads)
    for t in 1 2 4 6 8 10; do
        for f in email-Enron soc-Epinions1; do
            OMP_NUM_THREADS=$t ./dats -g data/$f.txt -snap -name "$f-omp$t" -only cpu -noserial -cpu 10 -csv results/cpu_threads.csv
        done
    done
    # 2. scaling with graph size (power-law graphs, avg degree 8)
    for n in 5000 10000 20000 50000 100000 200000; do
        ./dats -gen $n 8 -name "pl-$n" -only cpu -cpu 4 -csv results/cpu_scaling.csv
    done
    # 3. tile size (CPU only, GPU needs 32)
    for T in 8 16 32 64 128; do
        build_cpu dats_t$T "-DTILE_SIZE=$T"
        for f in wiki-Vote email-Enron soc-Epinions1; do
            ./dats_t$T -g data/$f.txt -snap -name "$f-T$T" -only cpu -cpu 4 -csv results/cpu_tilesize.csv
        done
        ./dats_t$T -g data/G55.txt -name "G55-T$T" -only cpu -cpu 4 -csv results/cpu_tilesize.csv
        rm -f dats_t$T
    done
    # 4. solution quality vs number of steps
    for s in 1000 3000 10000 30000 100000; do
        for g in G22 G55; do ./dats -g data/$g.txt -name "$g-s$s" -only cpu -noserial -cpu 10 -steps $s -csv results/convergence.csv; done
        ./dats -g data/email-Enron.txt -snap -name "email-Enron-s$s" -only cpu -noserial -cpu 10 -steps $s -csv results/convergence.csv
    done
    rm -f dats
    echo "done, results are in results/"
    exit 0
fi

if [ "$MODE" == "gpu" ]; then
    nvcc -O3 -arch=sm_75 -x cu -Xcompiler -fopenmp -lgomp maxcut_research.cpp -o dats || exit 1
    OUT=results/gpu.csv
    rm -f $OUT
    nvidia-smi --query-gpu=name --format=csv,noheader > results/gpu_name.txt
    for g in G1 G14 G22 G43 G48 G55; do ./dats -g data/$g.txt -name $g -only gpu -csv $OUT; done
    for f in wiki-Vote email-Enron ca-AstroPh soc-Epinions1; do
        ./dats -g data/$f.txt -snap -name $f -only gpu -csv $OUT
        ./dats -g data/$f.txt -snap -name $f -only gpu -shuffle -csv $OUT
    done
    ./dats -gen 100000 8 -name powerlaw-100k -only gpu -csv $OUT
    # block size study on the biggest real graph
    for t in 64 128 256 512 1024; do
        ./dats -g data/soc-Epinions1.txt -snap -name "epinions-tpb$t" -only gpu -threads $t -csv results/gpu_threads.csv
    done
else
    if command -v nvcc > /dev/null; then
        nvcc -O3 -x cu -Xcompiler -fopenmp -lgomp maxcut_research.cpp -o dats || exit 1
    elif [ "$(uname)" == "Darwin" ]; then
        OMP=$(brew --prefix libomp)
        clang++ -std=c++17 -O3 -Xpreprocessor -fopenmp -I$OMP/include -L$OMP/lib -lomp maxcut_research.cpp -o dats || exit 1
    else
        g++ -O3 -fopenmp maxcut_research.cpp -o dats || exit 1
    fi
    OUT=results/cpu.csv
    rm -f $OUT results/quality.csv
    sysctl -n machdep.cpu.brand_string > results/cpu_name.txt 2> /dev/null || grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 > results/cpu_name.txt
    for g in G1 G14 G22 G43 G48 G55; do ./dats -g data/$g.txt -name $g -only cpu -csv $OUT; done
    for f in wiki-Vote email-Enron ca-AstroPh soc-Epinions1; do
        ./dats -g data/$f.txt -snap -name $f -only cpu -csv $OUT
        ./dats -g data/$f.txt -snap -name $f -only cpu -shuffle -csv $OUT
    done
    ./dats -gen 100000 8 -name powerlaw-100k -only cpu -csv $OUT
    # longer runs to check solution quality against best known G-set values
    for g in G1 G14 G22 G43 G48 G55; do
        ./dats -g data/$g.txt -name $g -only cpu -noserial -steps 100000 -cpu 10 -csv results/quality.csv
    done
fi
echo "done, results are in results/"
