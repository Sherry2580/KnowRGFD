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
    def __init__(self, in_dim, hidden_dim, num_relations, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            self.layers.append(RGCNConv(in_c, hidden_dim, num_relations))

    def forward(self, x, edge_index, edge_type):
        hiddens = []
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            hiddens.append(x)
        return hiddens  # [N, hidden_dim] * num_layers

class KnowRGFD(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.L_RGCN = RGCNBlock(
            in_dim=dataset.num_features,
            hidden_dim=args.hidden,
            num_relations=args.num_relations,
            num_layers=args.local_layers
        )
        self.G_RGCN = RGCNBlock(
            in_dim=dataset.num_features,
            hidden_dim=args.hidden,
            num_relations=args.num_relations,
            num_layers=args.global_layers
        )
        self.local_layers = args.local_layers
        self.global_layers = args.global_layers
        self.num_classes = dataset.num_classes
        # node-wise attention/gating
        self.local_attn_net = nn.Linear(args.hidden, self.local_layers)
        self.global_attn_net = nn.Linear(args.hidden, self.global_layers)
        self.local_out_proj = nn.Linear(args.hidden, self.num_classes)
        self.global_out_proj = nn.Linear(args.hidden, self.num_classes)

    def layer_attention(self, hiddens, attn_score):
        # hiddens: List[Tensor] (len=L, 每個 shape: [N, C])
        h_stack = torch.stack(hiddens, dim=1)  # [N, L, C]
        # attn_score: [N, L]
        h_weighted = h_stack * attn_score[:, :, None]  # [N, L, C]
        h_final = h_weighted.sum(dim=1)  # [N, C]
        return h_final

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        edge_index = edge_index.to(x.device)
        edge_type = edge_type.to(x.device)
        local_hiddens = self.L_RGCN(x, edge_index, edge_type)
        global_hiddens = self.G_RGCN(x, edge_index, edge_type)
        # 用最後一層 hidden 當作 node feature 產生 gating
        local_attn_score = torch.softmax(self.local_attn_net(local_hiddens[-1]), dim=1)  # [N, L]
        global_attn_score = torch.softmax(self.global_attn_net(global_hiddens[-1]), dim=1)  # [N, L]
        local_feat = self.layer_attention(local_hiddens, local_attn_score)
        global_feat = self.layer_attention(global_hiddens, global_attn_score)
        local_out = self.local_out_proj(local_feat)
        global_out = self.global_out_proj(global_feat)
        prob_local = F.log_softmax(local_out, dim=1)
        prob_global = F.log_softmax(global_out, dim=1)
        return local_out, global_out, prob_local, global_out, global_out, prob_global

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

            # 新增 gating diversity regularization (KL)
            # 取得 gating 分布
            with torch.no_grad():
                x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
                edge_index = edge_index.to(x.device)
                edge_type = edge_type.to(x.device)
                local_hiddens = model.L_RGCN(x, edge_index, edge_type)
                global_hiddens = model.G_RGCN(x, edge_index, edge_type)
                local_attn_score = torch.softmax(model.local_attn_net(local_hiddens[-1]), dim=1)  # [N, local_layers]
                global_attn_score = torch.softmax(model.global_attn_net(global_hiddens[-1]), dim=1)  # [N, global_layers]
            # 只對重疊層（local 層數）做 KL
            min_layers = min(local_attn_score.shape[1], global_attn_score.shape[1])
            local_attn_overlap = local_attn_score[:, :min_layers]
            global_attn_overlap = global_attn_score[:, :min_layers]
            gating_kl = -(F.kl_div(local_attn_overlap.log(), global_attn_overlap, reduction='batchmean') + \
                        F.kl_div(global_attn_overlap.log(), local_attn_overlap, reduction='batchmean'))
            loss += getattr(args, 'lambda_gating_div', 0.1) * gating_kl

            # 檢查 loss 是否為 NaN
            if torch.isnan(loss):
                print(f"[ERROR] Epoch {epoch}: Loss became NaN!")
                print("log_prob_local:", log_prob_local)
                print("log_prob_global:", log_prob_global)
                print("labels:", labels)
                print("loss_ce:", loss_ce)
                print("loss_cr:", loss_cr)
                print("loss_gd:", gating_kl)
                print("weights:", weights)
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
                print(f"Epoch {epoch}, CE loss: {loss_ce.item():.4f}, CR loss: {loss_cr.item():.4f}, GD loss: {gating_kl.item():.4f}")
                print('Train acc: {:.4f}, Val acc: {:.4f}, Test acc: {:.4f}'.format(
                    train_metrics.metrics['accs'][f'fold{fold+1}'][-1],
                    val_metrics.metrics['accs'][f'fold{fold+1}'][-1],
                    test_metrics.metrics['accs'][f'fold{fold+1}'][-1]
                ))

            pbar.set_postfix({'loss': loss.detach().cpu().item()})
            pbar.update()

            # 新增：print 層級 attention/gating 權重
            if epoch % 10 == 0 or epoch == 1:
                # 取得 forward 時的 gating 分布
                with torch.no_grad():
                    x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
                    edge_index = edge_index.to(x.device)
                    edge_type = edge_type.to(x.device)
                    local_hiddens = model.L_RGCN(x, edge_index, edge_type)
                    global_hiddens = model.G_RGCN(x, edge_index, edge_type)
                    local_attn_score = torch.softmax(model.local_attn_net(local_hiddens[-1]), dim=1)  # [N, L]
                    global_attn_score = torch.softmax(model.global_attn_net(global_hiddens[-1]), dim=1)  # [N, G]
                    # print 前 3 個 node 的 gating 分布
                    print(f"[Epoch {epoch}] local_attn (node 0): {local_attn_score[0].cpu().numpy()}")
                    print(f"[Epoch {epoch}] local_attn (node 1): {local_attn_score[1].cpu().numpy()}")
                    print(f"[Epoch {epoch}] local_attn (node 2): {local_attn_score[2].cpu().numpy()}")
                    print(f"[Epoch {epoch}] global_attn (node 0): {global_attn_score[0].cpu().numpy()}")
                    print(f"[Epoch {epoch}] global_attn (node 1): {global_attn_score[1].cpu().numpy()}")
                    print(f"[Epoch {epoch}] global_attn (node 2): {global_attn_score[2].cpu().numpy()}")
    
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
    epsilon = 1e-8
    ps = [torch.exp(p).clamp(min=epsilon) for p in logps]
    sum_p = ps[0] + lambda_g * ps[1]
    avg_p = sum_p / len(ps)
    avg_p = avg_p.clamp(min=epsilon)
    sharp_p = (torch.pow(avg_p, 1./tem) / torch.sum(torch.pow(avg_p, 1./tem), dim=1, keepdim=True)).detach()
    loss = 0.
    for i, p in enumerate(ps):
        if cr_loss == 'kl':
            log_p = torch.log(p.clamp(min=epsilon))
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
                loss += torch.mean((p - sharp_p).pow(2).sum(1)[avg_p.max(1)[0] > conf])
            else:
                loss += (w[avg_p.max(1)[0] > conf] * (p - sharp_p).pow(2).sum(1)[avg_p.max(1)[0] > conf]).mean()
        else:
            raise ValueError(f"Unknown loss type: {cr_loss}")
    loss = loss / len(ps)
    if torch.isnan(sharp_p).any() or torch.isnan(ps[0]).any() or torch.isnan(ps[1]).any():
        print("consis_loss nan debug:")
        print("sharp_p:", sharp_p)
        print("ps[0]:", ps[0])
        print("ps[1]:", ps[1])
    return loss
