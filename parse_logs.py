import os
import re
import pandas as pd
import argparse

def parse_logs(dataset):
    log_dir = "results"
    logs = [f for f in os.listdir(log_dir) if f.startswith(dataset) and f.endswith(".log")]

    results = []

    for log_file in logs:
        try:
            with open(os.path.join(log_dir, log_file), "r") as f:
                content = f.read()

            # 從檔名解析 lr 和 dropout 和 lambda_ce
            match_param = re.search(r'_lr([\d\.]+)_do([\d\.]+)_lce([\d\.]+)\.log', log_file)
            if match_param:
                lr = float(match_param.group(1))
                dropout = float(match_param.group(2))
                lambda_ce = float(match_param.group(3))
            # 如果沒有 lambda_ce，則只抓 lr 和 dropout
            else:
                match_param = re.search(r'_lr([\d\.]+)_do([\d\.]+)\.log', log_file)
                if match_param:
                    lr = float(match_param.group(1))
                    dropout = float(match_param.group(2))
                    lambda_ce = None
                else:
                    lr = dropout = lambda_ce = None

            # 抓 FINAL ACC
            match_acc = re.search(r'\[FINAL ACCS\]\s+([\d\.]+)', content)
            final_acc = float(match_acc.group(1)) if match_acc else None

            # 抓 FINAL F1S
            match_f1 = re.search(r'\[FINAL F1S\]\s+([\d\.]+)', content)
            final_f1 = float(match_f1.group(1)) if match_f1 else None

            results.append({
                "lr": lr,
                "dropout": dropout,
                "lambda_ce": lambda_ce,
                "final_acc": final_acc,
                "final_f1": final_f1
            })

        except Exception as e:
            print(f"Error processing {log_file}: {e}")

    df = pd.DataFrame(results)
    df = df.sort_values(by=["final_acc"], ascending=False)

    save_path = os.path.join(log_dir, f"{dataset}_hyper_search_results.csv")
    # 控制小數位數
    float_formatters = {
        'lr': '{:.4f}'.format,
        'dropout': '{:.1f}'.format,
        'lambda_ce': '{:.4f}'.format,
        'final_acc': '{:.4f}'.format,
        'final_f1': '{:.4f}'.format
    }
    df = df.style.format(float_formatters)
    # styler 要用 to_excel
    save_path = save_path.replace('.csv', '.xlsx')
    df.to_excel(save_path, index=False)

    print(f"Results saved to {save_path}")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="dataset name prefix")
    args = parser.parse_args()

    parse_logs(args.dataset)
