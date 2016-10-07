#!/bin/bash -u


# types of test
algorithms=("cubic"  "reno")
# types of perturbation
perturbation=()
perturbations+=('none')
perturbations+=('loss')
perturbations+=('delay')
perturbations+=('corruption')

# arg1: algorithm
# must explicitly specify to avoid accidents
if [[ ! -z ${1:-} ]]; then
  algorithms=($1)
fi
# arg2: perturbation
# must explicitly specify to avoid accidents
echo "-I-: perturbation types: 'none' 'delay' 'loss' 'corruption'"
if [[ ! -z ${2:-} ]]; then
  perturbations=($2)
fi

for perturbation in ${perturbations[@]}; do
  for algorithm in ${algorithms[@]}; do
    #out_file=results_${algorithm}_${perturbation}_${curtime}_${filenum}.json
    filenum=001
    trials=10
    for i in $(seq 1 1 $trials); do
      filenum=$(printf "%03d" $i)
      out_file=results_${algorithm}_${perturbation}_${filenum}.png
      ls $out_file
      #    continue
      if [[ $? -eq 0 ]]; then
        eog $out_file &
      fi
    done
  done
done
