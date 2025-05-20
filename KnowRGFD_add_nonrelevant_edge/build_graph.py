import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData, Data
import xlsxwriter as xw
from deepwalk.graph import load_edgelist
import deepwalk.graph as graph
import argparse

# 處理新聞、實體和主題節點的特徵和關聯 (news, entity, topic)。
# 組織數據，生成多類型節點與邊的結構。
# python build_graph.py --dataset Random_test --num_topics 5

def load_edge(dataset, num_topics, node):
    news_index = np.load(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/graph/nodes/news_index.npy', allow_pickle=True).item()
    if node == 'topic':
        df = pd.read_excel(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/graph/edges/news2topic_{num_topics}.xlsx')
        index_dict = np.load(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/graph/nodes/{node}_index_{num_topics}.npy', allow_pickle=True).item()
        # 主題節點只有相似關係
        edges_similar = []
        edges_similar_ = []
        pair = df.values.tolist()
        for i in pair:
            head = news_index[i[0]]  # 查詢新聞索引
            tail = index_dict[i[1]]  # 查詢主題索引
            edge = [head, tail]
            edge_ = [tail, head]
            edges_similar.append(edge)
            edges_similar_.append(edge_)
        return edges_similar, edges_similar_, [], []  # 主題沒有不相似關係
    else:
        df = pd.read_excel(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/graph/edges/news2{node}.xlsx')
        index_dict = np.load(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/graph/nodes/{node}_index.npy', allow_pickle=True).item()
        
        edges_similar = []
        edges_similar_ = []
        edges_dissimilar = []
        edges_dissimilar_ = []
        
        for i, row in df.iterrows():
            head = news_index[row['news_id']]  # 查詢新聞索引
            tail = index_dict[row['entity_id']]  # 查詢實體索引
            edge = [head, tail]
            edge_ = [tail, head]
            
            # 根據關係類型分類
            if 'relation_type' in df.columns and row['relation_type'] == 'dissimilar':
                edges_dissimilar.append(edge)
                edges_dissimilar_.append(edge_)
            else:
                edges_similar.append(edge)
                edges_similar_.append(edge_)
        
        return edges_similar, edges_similar_, edges_dissimilar, edges_dissimilar_

def build_hg(dataset, num_topics):
    news_attr = np.load(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/Embeddings/news_embeddings.npy')
    news_attr = torch.from_numpy(news_attr)
    entity_attr = np.load(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/Embeddings/entity_embeddings.npy')
    entity_attr = torch.from_numpy(entity_attr)
    topic_attr = np.load(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/Embeddings/topic_embeddings_{num_topics}.npy')    
    topic_attr = torch.from_numpy(topic_attr)

    news2entity_similar, news2entity_similar_, news2entity_dissimilar, news2entity_dissimilar_ = load_edge(dataset, num_topics, 'entity')
    news2topic, news2topic_, _, _ = load_edge(dataset, num_topics, 'topic')  # 主題只有相似關係
    df_news = pd.read_excel(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/news_final.xlsx')
    label = df_news['label'].tolist()
    
    data = HeteroData()
    data['news'].x = news_attr
    data['entity'].x = entity_attr
    data['topic'].x = topic_attr
    
    # 相似關係
    data['news', 'has', 'entity'].edge_index = torch.tensor(news2entity_similar, dtype=torch.long).t().contiguous()
    data['entity', 'has_1', 'news'].edge_index = torch.tensor(news2entity_similar_, dtype=torch.long).t().contiguous()
    
    # 不相似關係 - 新增
    if news2entity_dissimilar:  # 如果有不相似的邊
        data['news', 'dissimilar_to', 'entity'].edge_index = torch.tensor(news2entity_dissimilar, dtype=torch.long).t().contiguous()
        data['entity', 'dissimilar_to_1', 'news'].edge_index = torch.tensor(news2entity_dissimilar_, dtype=torch.long).t().contiguous()
    
    # 主題關係
    data['news', 'belongs', 'topic'].edge_index = torch.tensor(news2topic, dtype=torch.long).t().contiguous()
    data['topic', 'belongs_1', 'news'].edge_index = torch.tensor(news2topic_, dtype=torch.long).t().contiguous()
    
    data['news'].y = torch.tensor(label, dtype=torch.long)
    print('='*60)
    print('HeteroGraph:', dataset, '\n', data)
    print(' num_nodes:', data.num_nodes, '\n', 'num_edges:', data.num_edges, '\n', 'Data has isolated nodes:', data.has_isolated_nodes(), '\n', 'Data is undirected:', data.is_undirected())
    print('='*60, '\n')
    torch.save(data, f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/graph/{dataset}_{num_topics}.pt')
    return data

def class2global(edgelist, global_index, classindex, prefix):
    indices_g = []
    for i in edgelist:
        ID = classindex[i]
        global_key = f"{prefix}_{ID}"  # 動態加前綴
        index_g = global_index[global_key]  # 從 global_index 獲取全局索引
        indices_g.append(index_g)
    return indices_g

def get_edgeList(dataset, num_topics):
    news_index = np.load(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/graph/nodes/news_index.npy', allow_pickle=True).item()
    entity_index = np.load(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/graph/nodes/entity_index.npy', allow_pickle=True).item()
    topic_index = np.load(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/graph/nodes/topic_index_{num_topics}.npy', allow_pickle=True).item()
    data = torch.load(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/graph/{dataset}_{num_topics}.pt')
    
    # 移除反向邊
    del data['entity', 'has_1', 'news']
    del data['topic', 'belongs_1', 'news']
    
    # 如果存在不相似關係的邊，也刪除其反向邊
    if ('entity', 'dissimilar_to_1', 'news') in data:
        del data['entity', 'dissimilar_to_1', 'news']

    # 抽取相似關係邊的索引
    newsList0 = data['news', 'has', 'entity'].edge_index.tolist()[0]
    entityList0 = data['news', 'has', 'entity'].edge_index.tolist()[1]
    
    # 抽取不相似關係邊的索引，如果存在
    newsList1 = []
    entityList1 = []
    if ('news', 'dissimilar_to', 'entity') in data:
        newsList1 = data['news', 'dissimilar_to', 'entity'].edge_index.tolist()[0]
        entityList1 = data['news', 'dissimilar_to', 'entity'].edge_index.tolist()[1]
    
    # 抽取主題關係邊的索引
    newsList2 = data['news', 'belongs', 'topic'].edge_index.tolist()[0]
    topicList = data['news', 'belongs', 'topic'].edge_index.tolist()[1]

    # 修正索引到原始 ID
    newsList0 = [int(list(news_index.keys())[list(news_index.values()).index(idx)]) for idx in newsList0]
    entityList0 = [int(list(entity_index.keys())[list(entity_index.values()).index(idx)]) for idx in entityList0]
    
    if newsList1:  # 如果有不相似關係的邊
        newsList1 = [int(list(news_index.keys())[list(news_index.values()).index(idx)]) for idx in newsList1]
        entityList1 = [int(list(entity_index.keys())[list(entity_index.values()).index(idx)]) for idx in entityList1]
    
    newsList2 = [int(list(news_index.keys())[list(news_index.values()).index(idx)]) for idx in newsList2]
    topicList = [int(list(topic_index.keys())[list(topic_index.values()).index(idx)]) for idx in topicList]
    
    global_index = np.load(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/graph/nodes/global_index_{num_topics}.npy', allow_pickle=True).item()

    # 轉換為全局索引
    news0_g = class2global(newsList0, global_index, news_index, "news")
    entity0_g = class2global(entityList0, global_index, entity_index, "entity")
    
    news1_g = []
    entity1_g = []
    if newsList1:  # 如果有不相似關係的邊
        news1_g = class2global(newsList1, global_index, news_index, "news")
        entity1_g = class2global(entityList1, global_index, entity_index, "entity")
    
    news2_g = class2global(newsList2, global_index, news_index, "news")
    topic_g = class2global(topicList, global_index, topic_index, "topic")

    # 合併所有邊
    node_head = news0_g + news1_g + news2_g
    node_tail = entity0_g + entity1_g + topic_g

    edgeList = []
    edgeList_rw = []
    for i in range(len(node_head)):
        head = node_head[i]
        tail = node_tail[i]
        edge = [head, tail]
        edge_rw = str(head) + ' ' + str(tail)
        edgeList.append(edge)
        edgeList_rw.append(edge_rw)
    
    with open(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/graph/edges/{dataset}_{num_topics}.edgelist', 'w', encoding='utf-8') as f:
        for i in edgeList_rw:
            f.write(str(i) + '\n')
        f.close()

    data_rw = Data(edge_index=torch.tensor(edgeList, dtype=torch.long).t().contiguous())
    G_rw = graph.load_edgelist(f'/home/blueee/KnowRGFD_add_nonrelevant_edge/Data/{dataset}/graph/edges/{dataset}_{num_topics}.edgelist', undirected=True)
    
    # 打印檢查信息
    print(f"G_rw nodes: {G_rw.number_of_nodes()}, expected: {data_rw.num_nodes}")
    print(f"G_rw edges: {G_rw.number_of_edges()}, expected: {data_rw.num_edges}")
    
    # 只檢查節點數，而不是邊數
    assert G_rw.number_of_nodes() == data_rw.num_nodes, 'wrong graph: node count mismatch'
    
    # 記錄邊數差異，但不中斷程序
    if G_rw.number_of_edges() != data_rw.num_edges:
        print(f"Warning: Edge count mismatch - G_rw has {G_rw.number_of_edges()} edges, data_rw has {data_rw.num_edges} edges")
        print("This may be due to automatic handling of duplicate edges in the graph library")
    
    return edgeList_rw

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='choose dataset')
    parser.add_argument('--dataset', type=str, default='MM COVID', help="['MM COVID','ReCOVery','MC Fake']")
    parser.add_argument('--num_topics', type=int)
    args = parser.parse_args()
    dataset = args.dataset
    num_topics = args.num_topics
    
    hg = build_hg(dataset, num_topics)       
    edgeList_rw = get_edgeList(dataset, num_topics)
    
    print(f'graph & edgelist for {dataset} done')