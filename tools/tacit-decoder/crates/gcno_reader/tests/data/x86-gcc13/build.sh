gcc sort.c -o sort.bin -ftest-coverage -g
gcc sort.c -o sort_cov.bin -fprofile-arcs -ftest-coverage -g

gcov-dump -l sort.bin-sort.gcno > sort.bin-sort.gcno.gcovdump
gcov-dump -l sort_cov.bin-sort.gcno > sort_cov.bin-sort.gcno.gcovdump

objdump -d sort.bin > sort.bin.objdump
objdump -d sort_cov.bin > sort_cov.bin.objdump

echo "---running sort.bin---"
./sort.bin
echo "---running sort_cov.bin---"
./sort_cov.bin

gcov-dump -l sort_cov.bin-sort.gcda > sort_cov.bin-sort.gcda.gcovdump

gcov sort_cov.bin-sort.gcno
