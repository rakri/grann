for M in 16 32 48 64;
do
	for ef in 100 200 400 800;
	do
		echo $M,$ef
		python3 hnsw_build.py $M $ef >> ~/hnsw_bench.log
	done
done

