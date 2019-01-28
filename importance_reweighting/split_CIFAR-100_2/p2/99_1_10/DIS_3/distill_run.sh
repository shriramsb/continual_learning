old_new=20.0
epsilons=(0.5 1.0)
sigma=10.0
num_repeat_expt=1
T=5.0
alpha=0.2

for epsilon in "${epsilons[@]}";
do
	echo "epsilon" $epsilon;
	echo "sigma" $sigma;
	python3 code_0_distill.py $old_new $epsilon $sigma $num_repeat_expt $T $alpha;
done
