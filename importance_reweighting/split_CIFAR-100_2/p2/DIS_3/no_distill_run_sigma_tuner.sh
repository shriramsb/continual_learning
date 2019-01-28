old_new=5.0
epsilon=0.5
sigmas=(2.0 5.0 7.0 15.0 20.0)
num_repeat_expt=10

for sigma in "${sigmas[@]}";
do
	echo "sigma" $sigma;
	python3 code_99_1_0_no_distill.py $old_new $epsilon $sigma $num_repeat_expt;
done
