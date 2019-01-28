old_new=5.0
epsilons=(0.0 0.2 0.5 1.0)
sigma=10.0
alphas=(0.0 0.1 0.2 0.7 1.0)
num_repeat_expt=10
T=5

for alpha in "${alphas[@]}";
do
	for epsilon in "${epsilons[@]}";
	do
		echo "alpha" $alpha;
		echo "epsilon" $epsilon;
		python3 code_99_1_0_distill.py $old_new $epsilon $sigma $num_repeat_expt $T $alpha;
	done
done
