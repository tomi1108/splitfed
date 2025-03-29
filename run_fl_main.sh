# !/bin/bash

anaconda_env=SL
model_type='resnet50' # ['mobilenet_v2, 'resnet50']
datasets_dir='~/datasets/' # path to datasets directory
results_dir='./results_linear/' # path to results directory
data_dist_type='non-iid' # ['iid', 'non-iid']

app_names=('FL') # ['FL']
# dataset_type_list=('cifar10' 'cifar100' 'tinyimagenet')
# alpha_list=(1.0 0.6 0.2)
dataset_type_list=('cifar10')
alpha_list=(1.0)

seed=42
num_rounds=50
num_epochs=5 # [1 5 10 20 40]
num_clients=5 
batch_size=128 
lr=0.01
min_lr=0.00001
momentum=0.9
weight_decay=0.0001

save_flag=True # [True, False]

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${anaconda_env}

for dataset_type in "${dataset_type_list[@]}"; do

    for alpha in "${alpha_list[@]}"; do

        for app_name in "${app_names[@]}"; do

            echo "dataset: ${dataset_type}, alpha: ${alpha}"

            params="
                --app_name ${app_name}
                --model_type ${model_type}
                --dataset_type ${dataset_type}
                --datasets_dir ${datasets_dir}
                --results_dir ${results_dir}
                --data_dist_type ${data_dist_type}
                --num_clients ${num_clients}
                --num_rounds ${num_rounds}
                --num_epochs ${num_epochs}
                --batch_size ${batch_size}
                --alpha ${alpha}
                --lr ${lr}
                --min_lr ${min_lr}
                --momentum ${momentum}
                --weight_decay ${weight_decay}
                --save_flag ${save_flag}
            "

            python3 fedavg.py ${params}
        
        done
    done
done