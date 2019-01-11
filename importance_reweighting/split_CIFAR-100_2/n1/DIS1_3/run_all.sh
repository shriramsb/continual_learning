python3 code_all.py 0 5 0 1 &
python3 code_all.py 1 5 1 1 &
wait
python3 code_test_acc.py 0 5 0 &
python3 code_test_acc.py 1 5 1 &
wait

python3 code_all.py 0 5 2 1 &
python3 code_all.py 1 5 3 1 &
wait
python3 code_test_acc.py 0 5 2 &
python3 code_test_acc.py 1 5 3 &
wait

python3 code_all.py 0 5 4 1 &
wait
python3 code_test_acc.py 0 5 4 &
wait