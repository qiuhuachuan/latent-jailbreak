prompt_types=('P2' 'P3' 'P4' 'P5' 'P6' 'P7' 'P8' 'P9' 'P10' 'P11' 'P12' 'P13')

for prompt_type in ${prompt_types[*]}; do
    python automatic_evaluation.py --prompt_type ${prompt_type}
done