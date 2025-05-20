import torch
from torch_geometric.data import  Data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import numpy as np

# 將異構圖轉換為同構圖並加入 edge_type（供 RGCN 使用）
def load_data(args):
    graph = torch.load(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{args.dataset}/graph/{args.dataset}_{args.num_topics}.pt')

    news_x = graph['news'].x
    entity_x = graph['entity'].x
    topic_x = graph['topic'].x
    labels = graph['news'].y

    # 拼接所有節點特徵
    x = torch.cat([news_x, entity_x, topic_x], dim=0).float()

    # 根據節點類型做 index 偏移
    n_news = news_x.size(0)
    n_entity = entity_x.size(0)
    n_topic = topic_x.size(0)

    # 各種邊及其類型
    edges = []
    edge_types = []

    # news -> entity (相似關係，type 0)
    e = graph['news', 'has', 'entity'].edge_index.clone()
    e[1] += n_news
    edges.append(e)
    edge_types.append(torch.full((e.size(1),), 0))

    # entity -> news (相似關係，type 1)
    e = graph['entity', 'has_1', 'news'].edge_index.clone()
    e[0] += n_news
    edges.append(e)
    edge_types.append(torch.full((e.size(1),), 1))

    # news -> entity (不相似關係，type 2)
    e = graph['news', 'dissimilar_to', 'entity'].edge_index.clone()
    e[1] += n_news
    edges.append(e)
    edge_types.append(torch.full((e.size(1),), 2))

    # entity -> news (不相似關係，type 3)
    e = graph['entity', 'dissimilar_to_1', 'news'].edge_index.clone()
    e[0] += n_news
    edges.append(e)
    edge_types.append(torch.full((e.size(1),), 3))

    # news -> topic (type 4)
    e = graph['news', 'belongs', 'topic'].edge_index.clone()
    e[1] += n_news + n_entity
    edges.append(e)
    edge_types.append(torch.full((e.size(1),), 4))

    # topic -> news (type 5)
    e = graph['topic', 'belongs_1', 'news'].edge_index.clone()
    e[0] += n_news + n_entity
    edges.append(e)
    edge_types.append(torch.full((e.size(1),), 5))

    # 合併所有邊與邊類型
    edge_index = torch.cat(edges, dim=1)
    edge_type = torch.cat(edge_types, dim=0)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=labels,
        n_news=n_news,
        num_classes=2
    )
    return data


def plot_gating_histogram(gating_score, layer_name="Layer", save_path=None):
    # gating_score: [N, L]，每個 node 的 softmax 分布
    num_layers = gating_score.shape[1]
    plt.figure(figsize=(8, 6))
    # flatten 所有 node 的 gating 分布，畫 histogram
    for l in range(num_layers):
        plt.hist(gating_score[:, l], bins=30, alpha=0.5, label=f'{layer_name} {l+1}')
    plt.xlabel('Gating Weight')
    plt.ylabel('Node Count')
    plt.title(f'Gating Distribution per {layer_name}')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
