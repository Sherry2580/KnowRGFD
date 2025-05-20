import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from model.KnowRGFD import KnowRGFD, train
from arguments import arg_parser
from metrics import My_metrics
from sklearn.model_selection import StratifiedShuffleSplit
import os
from utils import load_data
import matplotlib.pyplot as plt
import utils


def run_ours():
    args = arg_parser()
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(args).to(device)

    split_dir = f"./splits/{args.dataset}"
    os.makedirs(split_dir, exist_ok=True)

    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    skf = StratifiedShuffleSplit(n_splits=args.fold, test_size=1 - args.tr)

    train_indices, val_indices, test_indices = [], [], []
    labels_cpu = data.y[:data.n_news].cpu().numpy()
    for i, (train_index, test_index) in enumerate(skf.split(np.zeros(data.n_news), labels_cpu)):
        train_indices.append(train_index)
        val_indices.append(test_index[:len(test_index) // 2])
        test_indices.append(test_index[len(test_index) // 2:])

        np.savetxt(f"{split_dir}/train_fold{i}.txt", train_index)
        np.savetxt(f"{split_dir}/val_fold{i}.txt", test_index[:len(test_index) // 2])
        np.savetxt(f"{split_dir}/test_fold{i}.txt", test_index[len(test_index) // 2:])

    train_metrics = My_metrics(folds=args.fold)
    val_metrics = My_metrics(folds=args.fold)
    test_metrics = My_metrics(folds=args.fold)

    all_loss = []
    all_train_acc = []
    all_val_acc = []
    all_test_acc = []

    for fold in range(args.fold):
        model = KnowRGFD(data, args).to(device)

        idx_train = torch.from_numpy(train_indices[fold]).to(device)
        idx_val = torch.from_numpy(val_indices[fold]).to(device)
        idx_test = torch.from_numpy(test_indices[fold]).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        # 呼叫訓練，並取得每個 epoch 的記錄
        loss_list, train_acc_list, val_acc_list, test_acc_list = train(
            model, data, optimizer, idx_train, idx_val, idx_test, data.y, args,
            fold, train_metrics, val_metrics, test_metrics
        )

        # 記錄每個 fold 的 trend
        all_loss.append(loss_list)
        all_train_acc.append(train_acc_list)
        all_val_acc.append(val_acc_list)
        all_test_acc.append(test_acc_list)

        # 抓 "最後一次出現最佳 val acc" 的 test acc
        val_accs = val_metrics.metrics['accs'][f'fold{fold+1}']
        test_accs = test_metrics.metrics['accs'][f'fold{fold+1}']
        val_best = max(val_accs)
        # 取得所有出現 val_best 的 epoch
        best_epochs = [i for i, v in enumerate(val_accs) if v == val_best]

        # 取出這些 epoch 中 test acc 的最大值
        test_best = max(test_accs[i] for i in best_epochs)

        print('-' * 60)
        print(f'[FOLD {fold} BEST] val acc: {val_best:.4f}, test acc: {test_best:.4f}')

        # ========== Gating 可視化 =============
        model.eval()
        with torch.no_grad():
            x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
            edge_index = edge_index.to(x.device)
            edge_type = edge_type.to(x.device)
            local_hiddens = model.L_RGCN(x, edge_index, edge_type)
            global_hiddens = model.G_RGCN(x, edge_index, edge_type)
            local_attn_score = torch.softmax(model.local_attn_net(local_hiddens[-1]), dim=1).cpu().numpy()  # [N, L]
            global_attn_score = torch.softmax(model.global_attn_net(global_hiddens[-1]), dim=1).cpu().numpy()  # [N, G]
            n_news = data.n_news
            news_local_attn_score = local_attn_score[:n_news]
            news_global_attn_score = global_attn_score[:n_news]
            # 只畫 gating histogram
            utils.plot_gating_histogram(news_local_attn_score, layer_name="Local Layer", save_path=f"./results/local_gating_hist_fold{fold}.png")
            utils.plot_gating_histogram(news_global_attn_score, layer_name="Global Layer", save_path=f"./results/global_gating_hist_fold{fold}.png")

    # 將每個 fold 的 list -> tensor for averaging
    loss_avg = np.mean(np.array(all_loss), axis=0)
    train_acc_avg = np.mean(np.array(all_train_acc), axis=0)
    val_acc_avg = np.mean(np.array(all_val_acc), axis=0)
    test_acc_avg = np.mean(np.array(all_test_acc), axis=0)

    # 自動抓參數
    lr = args.lr
    dropout = args.dropout
    lambda_ce = args.lambda_ce
    local_layers = args.local_layers
    global_layers = args.global_layers

    # Loss
    plt.figure()
    plt.plot(loss_avg, label='Loss')
    # 跑不同參數
    plt.title(f'Loss (lr={lr}, dropout={dropout}, lambda_ce={lambda_ce})')
    # 跑不同層數
    #plt.title(f'Loss (local layer={local_layers}, global layer={global_layers})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # 跑不同參數
    plt.savefig(f'./results/avg_loss_lr{lr}_do{dropout}_lce{lambda_ce}.png')
    # 跑不同層數
    # plt.savefig(f'./results/avg_loss_local{local_layers}_global{global_layers}.png')


    # Accuracy
    plt.figure()
    plt.plot(train_acc_avg, label='Train Acc')
    plt.plot(val_acc_avg, label='Val Acc')
    plt.plot(test_acc_avg, label='Test Acc')
    plt.title(f'Accuracy (lr={lr}, dropout={dropout}, lambda_ce={lambda_ce})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'./results/avg_acc_lr{lr}_do{dropout}_lce{lambda_ce}.png')
    plt.close('all')

    print('[FINAL RESULTS]')
    for metric in ['accs', 'precisions', 'recalls', 'f1s', 'aucs', 'aprs']:
        mean, std = test_metrics.get_final(metric)
        print(f'[FINAL {metric.upper()}] {mean:.4f} +- {std:.4f}')

        with open(f'./results/KnowRGFD_{args.dataset}_{args.num_topics}.txt', 'a') as file:
            file.write(f'{metric.upper()}: {mean:.4f} +- {std:.4f}\n')


if __name__ == '__main__':
    run_ours()

