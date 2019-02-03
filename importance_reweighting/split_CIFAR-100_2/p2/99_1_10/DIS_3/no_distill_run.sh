old_new=5.0
epsilons=(0.5 1.0)
sigmas=(10.0)
num_repeat_expt=3

for epsilon in "${epsilons[@]}";
do
	for sigma in "${sigmas[@]}";
	do
		echo "epsilon" $epsilon;
		echo "sigma" $sigma;
		python3 code_0_no_distill.py $old_new $epsilon $sigma $num_repeat_expt;
	done
done
