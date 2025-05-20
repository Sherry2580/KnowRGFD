#!/bin/bash

# 超參數組合
learning_rates=(0.0002 0.0003 0.0005)
dropouts=(0.5 0.7 0.8)
lambda_ces=(0.5 0.7)

dataset="Knowledge_more15_PolitiFact_nontrun"
num_topics=30
epochs=150

mkdir -p results

for lr in "${learning_rates[@]}"; do
    for dropout in "${dropouts[@]}"; do
        for lambda_ce in "${lambda_ces[@]}"; do
            out_dir="node-wise_PolitiFact_lr${lr}_dro${dropout}_lce${lambda_ce}"
            mkdir -p $out_dir
            log_name="results/node-wise_PolitiFact_lr${lr}_dro${dropout}_lce${lambda_ce}.log"

            echo "Running: lr=$lr, dropout=$dropout, lambda_ce=$lambda_ce"

            nohup python main.py --dataset $dataset --num_topics $num_topics --lr $lr --lambda_ce $lambda_ce --dropout $dropout --epochs $epochs > $log_name &
            wait

            # 移動 main.py 產生的所有圖片和檔案到 out_dir
            mv results/*.png $out_dir/ 2>/dev/null
            mv results/*.pt $out_dir/ 2>/dev/null
            mv results/*.txt $out_dir/ 2>/dev/null
        done
    done
done

echo "All runs completed! Now parsing logs..."

#整理結果
python parse_logs.py --dataset $dataset