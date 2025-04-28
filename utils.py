import torch
from torch_geometric.data import  Data

# 將異構圖轉換為同構圖並加入 edge_type（供 RGCN 使用）
def load_data(args):
    graph = torch.load(f'/home/blueee/KnowRGFD/Data/{args.dataset}/graph/{args.dataset}_{args.num_topics}.pt')

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

    # news -> entity (type 0)
    e = graph['news', 'has', 'entity'].edge_index.clone()
    e[1] += n_news
    edges.append(e)
    edge_types.append(torch.full((e.size(1),), 0))

    # entity -> news (type 1)
    e = graph['entity', 'has_1', 'news'].edge_index.clone()
    e[0] += n_news
    edges.append(e)
    edge_types.append(torch.full((e.size(1),), 1))

    # news -> topic (type 2)
    e = graph['news', 'belongs', 'topic'].edge_index.clone()
    e[1] += n_news + n_entity
    edges.append(e)
    edge_types.append(torch.full((e.size(1),), 2))

    # topic -> news (type 3)
    e = graph['topic', 'belongs_1', 'news'].edge_index.clone()
    e[0] += n_news + n_entity
    edges.append(e)
    edge_types.append(torch.full((e.size(1),), 3))

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
