#nvprof -f -o out_o0_b512.sql --profile-from-start off -- python punctuation_pyprof_o0.py
#nvprof -f -o out_o1_b512.sql --profile-from-start off -- python punctuation_pyprof_o1.py

python $PYPROF/parse/parse.py out_o0_b512.sql > out_o0_b512.dict
python $PYPROF/parse/parse.py out_o1_b512.sql > out_o1_b512.dict

python $PYPROF/prof/prof.py -w 130 out_o0_b512.dict > out_o0_b512.out
python $PYPROF/prof/prof.py -w 130 out_o1_b512.dict > out_o1_b512.out

sort -k8 -n -r out_o0_b512.out > out_o0_b512.sorted
sort -k8 -n -r out_o1_b512.out > out_o1_b512.sorted
