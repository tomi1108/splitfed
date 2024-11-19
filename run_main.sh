#!/bin/bash

anaconda_env=SL
# app_names=('P_SFL') # ['SFL', 'P_SFL', 'PM_SFL']
app_name='P_SFL'
model_type='mobilenet_v2' # ['mobilenet_v2, 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
dataset_type='cifar10' # ['cifar10', 'cifar100', 'tinyimagenet']
datasets_dir='~/datasets/'
results_dir='./results/'
data_dist_type='non-iid' # ['iid', 'non-iid']

seed=42
num_clients=5
num_rounds=100
num_epochs=5
batch_size=128
alpha_list=(0.2 1.0 0.6)
# alpha=0.2
threshold=0.1
# u=0.0
u_list=(10.0 5.0)
lr=0.01
momentum=0.9
weight_decay=0.0001

save_flag=True # [True, False]

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${anaconda_env}

for alpha in "${alpha_list[@]}"; do

    for u in "${u_list[@]}"; do

        if [[ "${alpha}" == "0.2" && "${u}" == "5.0" ]]; then
            echo "Skipping alpha=0.2 and u=5.0"
            continue
        fi

        echo "alpha: ${alpha}, u: ${u}"
        params="
            --app_name ${app_name}
            --model_type ${model_type}
            --dataset_type ${dataset_type}
            --datasets_dir ${datasets_dir}
            --data_dist_type ${data_dist_type}
            --num_clients ${num_clients}
            --num_rounds ${num_rounds}
            --num_epochs ${num_epochs}
            --batch_size ${batch_size}
            --alpha ${alpha}
            --threshold ${threshold}
            --u ${u}
            --lr ${lr}
            --momentum ${momentum}
            --weight_decay ${weight_decay}
            --save_flag ${save_flag}
        "

        python3 main.py ${params}

    done
done

# for app_name in "${app_names[@]}"; do
#     if [ "$app_name" == "SFL" ]; then
#         # `SFL` の場合は1度だけ実行
#         u=1.0
#         echo "Running with app_name=${app_name} and u=${u}"
#         params="
#             --app_name ${app_name}
#             --model_type ${model_type}
#             --dataset_type ${dataset_type}
#             --datasets_dir ${datasets_dir}
#             --data_dist_type ${data_dist_type}
#             --num_clients ${num_clients}
#             --num_rounds ${num_rounds}
#             --num_epochs ${num_epochs}
#             --batch_size ${batch_size}
#             --alpha ${alpha}
#             --threshold ${threshold}
#             --u ${u}
#             --lr ${lr}
#             --momentum ${momentum}
#             --weight_decay ${weight_decay}
#             --save_flag ${save_flag}
#         "
#         python3 main.py ${params}

#     elif [ "$app_name" == "P_SFL" ]; then
#         # `P_SFL` の場合はすべての `u` の値で実行
#         for u in "${u_list[@]}"; do
#             echo "Running with app_name=${app_name} and u=${u}"
#             params="
#                 --app_name ${app_name}
#                 --model_type ${model_type}
#                 --dataset_type ${dataset_type}
#                 --datasets_dir ${datasets_dir}
#                 --data_dist_type ${data_dist_type}
#                 --num_clients ${num_clients}
#                 --num_rounds ${num_rounds}
#                 --num_epochs ${num_epochs}
#                 --batch_size ${batch_size}
#                 --alpha ${alpha}
#                 --threshold ${threshold}
#                 --u ${u}
#                 --lr ${lr}
#                 --momentum ${momentum}
#                 --weight_decay ${weight_decay}
#                 --save_flag ${save_flag}
#             "
#             python3 main.py ${params}
#         done
#     fi
# done