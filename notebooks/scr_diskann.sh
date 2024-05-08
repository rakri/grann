for R in 16 32 48 64 96;
do
	for L in 100 200 300;
	do
		echo $R,$L
		~/DiskANN/build/apps/build_memory_index --data_type float --dist_fn l2 --data_path ~/wiki_rnd1m_data.bin --index_path_prefix ~/wiki_rnd1m_${R}_${L} -R ${R} -L ${L} >> ~/diskann_bench.log
		~/DiskANN/build/apps/search_memory_index --data_type float --dist_fn l2 --index_path_prefix ~/wiki_rnd1m_${R}_${L} --query_file ~/wiki_rnd1m/wikipedia_query.bin --gt_file ~/wiki_rnd1m_gt100.bin --result_path /tmp/res -K 10 -L 10 20 30 40 50 60 70 80 90 -T 1 >> ~/diskann_bench.log
	done
done

