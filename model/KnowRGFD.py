import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
import torch.nn.init as init
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, roc_curve, auc
from torch_geometric.nn import RGCNConv


class RGCNBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            out_c = hidden_dim if i < num_layers - 1 else out_dim
            self.layers.append(RGCNConv(in_c, out_c, num_relations))

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index, edge_type)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

class KnowRGFD(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.L_RGCN = RGCNBlock(
            in_dim=dataset.num_features,
            hidden_dim=args.hidden,
            out_dim=dataset.num_classes,
            num_relations=args.num_relations,
            num_layers=2  # local 視圖
        )
        self.G_RGCN = RGCNBlock(
            in_dim=dataset.num_features,
            hidden_dim=args.hidden,
            out_dim=dataset.num_classes,
            num_relations=args.num_relations,
            num_layers=4  # global 視圖
        )

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        # ✅ 保證都在相同 device
        edge_index = edge_index.to(x.device)
        edge_type = edge_type.to(x.device)

        logits_local = self.L_RGCN(x, edge_index, edge_type)
        logits_global = self.G_RGCN(x, edge_index, edge_type)
        prob_local = F.log_softmax(logits_local, dim=1)
        prob_global = F.log_softmax(logits_global, dim=1)
        return logits_local, logits_global, prob_local, logits_global, logits_global, prob_global

def train(model, data, optimizer, idx_train, idx_val, idx_test, labels, args, 
          fold, train_metrics, val_metrics, test_metrics):
    loss_list = []
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    unsup_idx = torch.cat((idx_val, idx_test)).to(data.x.device)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    labels = labels.to(device).long()

    with tqdm(total=args.epochs, desc='(Training KnowRGFD)', disable=not args.verbose) as pbar:
        for epoch in range(1, 1+args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = 0

            # 模型輸出
            h1_l, h2_l, log_prob_local, h1_g, h2_g, log_prob_global = model(data)

            # 計算類別不平衡的權重
            num_fake = data.y.sum().double()
            num_real = data.y.shape[0] - num_fake
            weights = torch.tensor([1., num_real / num_fake], dtype=torch.float32, device=device)

            # 計算 CrossEntropy Loss
            loss_ce = F.nll_loss(log_prob_local[idx_train], labels[idx_train], weight=weights) + \
                      args.lambda_g * F.nll_loss(log_prob_global[idx_train], labels[idx_train], weight=weights)
            loss += args.lambda_ce * loss_ce

            # 一致性損失
            lambda_cr = 1 - args.lambda_ce
            if lambda_cr > 0:
                if args.onlyUnlabel == "yes":
                    loss_cr = consis_loss(args.cr_loss, [log_prob_local[unsup_idx], log_prob_global[unsup_idx]], args.cr_tem, args.cr_conf, args.lambda_g)
                else:
                    loss_cr = consis_loss(args.cr_loss, [log_prob_local, log_prob_global], args.cr_tem, args.cr_conf, args.lambda_g)
                loss += lambda_cr * loss_cr
            else:
                loss_cr = torch.tensor(0.0, device=device)

            # 檢查 loss 是否為 NaN
            if torch.isnan(loss):
                print(f"[ERROR] Epoch {epoch}: Loss became NaN!")
                print("log_prob_local:", log_prob_local)
                print("log_prob_global:", log_prob_global)
                print("labels:", labels)
                return None

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 評估
            with torch.no_grad():
                model.eval()
                _, _, log_prob_local, _, _, log_prob_global = model(data)
                prob_local = torch.exp(log_prob_local)
                prob_global = torch.exp(log_prob_global)
                y_final = args.beta * prob_local + (1 - args.beta) * prob_global

                y_pred = y_final.argmax(dim=1)
                print(f"[Epoch {epoch}] Predicted label distribution:", torch.bincount(y_pred))

                auc_list1 = validation(y_final, labels, idx_train, train_metrics, fold)
                auc_list2 = validation(y_final, labels, idx_val, val_metrics, fold)
                auc_list3 = validation(y_final, labels, idx_test, test_metrics, fold)
                torch.save(auc_list3, f'./results/{args.dataset}_auc.pt')

                # 記錄 loss 和 acc
                loss_list.append(loss.item())
                train_acc = train_metrics.metrics['accs'][f'fold{fold+1}'][-1]
                val_acc = val_metrics.metrics['accs'][f'fold{fold+1}'][-1]
                test_acc = test_metrics.metrics['accs'][f'fold{fold+1}'][-1]

                train_acc_list.append(train_acc)
                val_acc_list.append(val_acc)
                test_acc_list.append(test_acc)
 
            
            if epoch % args.eval_freq == 0 and args.verbose:
                print(f"Epoch {epoch}, CE loss: {loss_ce.item():.4f}, CR loss: {loss_cr.item():.4f}")
                print('Train acc: {:.4f}, Val acc: {:.4f}, Test acc: {:.4f}'.format(
                    train_metrics.metrics['accs'][f'fold{fold+1}'][-1],
                    val_metrics.metrics['accs'][f'fold{fold+1}'][-1],
                    test_metrics.metrics['accs'][f'fold{fold+1}'][-1]
                ))

            pbar.set_postfix({'loss': loss.detach().cpu().item()})
            pbar.update()
    
    return loss_list, train_acc_list, val_acc_list, test_acc_list

def validation(y_final, labels, idx, metric, fold):
    y_pred = y_final.argmax(dim=-1, keepdim=True)[idx.cpu()]
    labels = labels[idx.cpu()]

    metric.metrics['accs'][f'fold{fold+1}'].append(accuracy_score(labels.cpu(), y_pred.cpu()))
    metric.metrics['precisions'][f'fold{fold+1}'].append(precision_score(labels.cpu(), y_pred.cpu(), average='macro'))
    metric.metrics['recalls'][f'fold{fold+1}'].append(recall_score(labels.cpu(), y_pred.cpu(), average='macro'))
    metric.metrics['f1s'][f'fold{fold+1}'].append(f1_score(labels.cpu(), y_pred.cpu(), average='macro'))
    metric.metrics['aucs'][f'fold{fold+1}'].append(roc_auc_score(labels.cpu(), y_pred.cpu(), average='macro'))
    
    y_test = y_final.cpu()[idx.cpu()]
    if torch.isnan(y_test).any():
        print(f"[WARNING] Fold {fold}: y_test contains NaN, skipping ROC curve")
        return [None, None, None, 0.0]

    fpr, tpr, thresholds = roc_curve(labels.cpu(), y_test[:, 1])
    roc_auc = auc(fpr, tpr)
    auc_list = [fpr, tpr, thresholds, roc_auc]

    metric.metrics['aprs'][f'fold{fold+1}'].append(average_precision_score(labels.cpu(), y_pred.cpu(), average='macro'))
    return auc_list

def consis_loss(cr_loss, logps, tem, conf, lambda_g, w=1.0, reduction='none'):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    sum_p = ps[0] + lambda_g * ps[1]
    avg_p = sum_p/len(ps)
    avg_p = sum_p/len(ps)

    sharp_p = (torch.pow(avg_p, 1./tem) / torch.sum(torch.pow(avg_p, 1./tem), dim=1, keepdim=True)).detach()
    loss = 0.
    for i,p in enumerate(ps):
        if cr_loss == 'kl':
            log_p = torch.log(p)
            kl_loss = -F.kl_div(log_p, sharp_p, reduction='none').sum(1)
            if reduction != 'none':
                filtered_kl_loss = kl_loss[avg_p.max(1)[0] > conf]
                loss += torch.mean(filtered_kl_loss)
            else:
                loss += (w * kl_loss).mean()
                if i == 1:
                    loss = loss * lambda_g

        elif cr_loss == 'l2':
            if reduction != 'none':
                loss +=  torch.mean((p-sharp_p).pow(2).sum(1)[avg_p.max(1)[0] > conf])
            else:
                loss += (w[avg_p.max(1)[0] > conf] * (p-sharp_p).pow(2).sum(1)[avg_p.max(1)[0] > conf]).mean()
        else:
            raise ValueError(f"Unknown loss type: {cr_loss}")
    loss = loss/len(ps)
    return loss
