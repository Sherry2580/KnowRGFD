#!/bin/bash

# 超參數組合
learning_rates=(0.0002 0.0003 0.0005)
dropouts=(0.8 0.85 0.9)
lambda_ces=(0.6 0.7)

dataset="Knowledge_more15_PolitiFact_nontrun"
num_topics=30
epochs=80

mkdir -p results

for lr in "${learning_rates[@]}"; do
    for dropout in "${dropouts[@]}"; do
        for lambda_ce in "${lambda_ces[@]}"; do
            log_name="results/${dataset}_lr${lr}_do${dropout}_lce${lambda_ce}.log"

            echo "Running: lr=$lr, dropout=$dropout, lambda_ce=$lambda_ce"

            nohup python main.py --dataset $dataset --num_topics $num_topics --lr $lr --lambda_ce $lambda_ce --dropout $dropout --epochs $epochs > $log_name &

            wait
        done
    done
done

echo "All runs completed! Now parsing logs..."

#整理結果
python parse_logs.py --dataset $dataset