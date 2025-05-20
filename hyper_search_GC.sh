#!/bin/bash

learning_rates=(0.0005 0.001 0.002)
dropouts=(0.5 0.6 0.7)

dataset="Knowledge_more15_GossipCop_nontrun"
num_topics=30
lambda_ce=0.8
epochs=200

mkdir -p results

for lr in "${learning_rates[@]}"; do
  for dropout in "${dropouts[@]}"; do
    log_name="results/${dataset}_lr${lr}_do${dropout}.log"

    echo "Running: lr=$lr, dropout=$dropout"
    
    nohup python main.py --dataset $dataset --num_topics $num_topics --lr $lr --lambda_ce $lambda_ce --dropout $dropout --epochs $epochs > $log_name &

    wait  # 等這組跑完才繼續下一組
  done
done

echo "All runs completed! Now parsing logs..."

# 整理log的程式
python parse_logs.py --dataset $dataset
