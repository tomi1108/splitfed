# !/bin/bash

anaconda_env=SL
model_type='resnet50' # ['mobilenet_v2, 'resnet50']
datasets_dir='~/datasets/' # path to datasets directory
results_dir='./results_linear/' # path to results directory
data_dist_type='non-iid' # ['iid', 'non-iid']

# app_names=('SFL' 'P_SFL') # ['SFL', 'P_SFL', 'PM_SFL']
# dataset_type_list=('cifar10' 'cifar100' 'tinyimagenet')
# alpha_list=(1.0 0.6 0.2)
# mu_list=(10.0 5.0 1.0)
# lambda_list=(0.9 0.5 0.1)

# app_names=('PKL_SFL') # ['SFL', 'P_SFL', 'PKL_SFL']
app_names=('MOON_SFL')
dataset_type_list=('cifar100') # ['cifar10', 'cifar100', 'tinyimagenet']
alpha_list=(0.2) # [1.0 0.6 0.2]
mu_list=(5.0) # [10.0 5.0 1.0]
lambda_list=(0.1) # [1.0 0.9 0.5 0.1]

seed=42
num_rounds=50
warmup_rounds=0
num_epochs=5 # [1 5 10 20 40]
num_clients=5 
batch_size=128 
projected_size=0 # CIFAR-10: 128, CIFAR-100: 256, Tiny-ImageNet: 512
lr=0.01
min_lr=0.00001
momentum=0.9
weight_decay=0.0001

save_flag=True # [True, False]

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${anaconda_env}

for dataset_type in "${dataset_type_list[@]}"; do

    for alpha in "${alpha_list[@]}"; do

        for mu in "${mu_list[@]}"; do

            for lambda in "${lambda_list[@]}"; do

                for app_name in "${app_names[@]}"; do

                    # if [[ "${app_name}" == 'P_SFL' && "${alpha}" == '0.2' && "${lambda}" == '1.0' ]]; then
                    #     continue
                    # fi

                    # if [[ "${app_name}" == 'PKL_SFL' && "${lambda}" == 0.5 && "${alpha}" == 0.2 && "${mu}" == 10.0 ]]; then
                    #     continue
                    # fi

                    # if [[ "${app_name}" == 'SFL' && ( "${lambda}" == 0.5 || "${lambda}" == 0.1 || "${mu}" == 5.0 || "${mu}" == 1.0 ) ]]; then
                    #     continue
                    # fi

                    # echo "approach name: ${app_name}"
                    # echo "alpha: ${alpha}"
                    # echo "mu: ${mu}"
                    # echo "lambda: ${lambda}"
                    echo "dataset: ${dataset_type}, alpha: ${alpha}, mu: ${mu}, lambda: ${lambda}"

                    params="
                        --app_name ${app_name}
                        --model_type ${model_type}
                        --dataset_type ${dataset_type}
                        --datasets_dir ${datasets_dir}
                        --results_dir ${results_dir}
                        --data_dist_type ${data_dist_type}
                        --num_clients ${num_clients}
                        --num_rounds ${num_rounds}
                        --warmup_rounds ${warmup_rounds}
                        --num_epochs ${num_epochs}
                        --projected_size ${projected_size}
                        --batch_size ${batch_size}
                        --alpha ${alpha}
                        --mu ${mu}
                        --lam ${lambda}
                        --lr ${lr}
                        --min_lr ${min_lr}
                        --momentum ${momentum}
                        --weight_decay ${weight_decay}
                        --save_flag ${save_flag}
                    "

                    python3 main.py ${params}
                
                done
            done
        done
    done
done