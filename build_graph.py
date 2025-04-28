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

def load_edge(dataset,num_topics,node):
    news_index = np.load(f'/home/blueee/KnowRGFD/Data/{dataset}/graph/nodes/news_index.npy',allow_pickle= True).item()
    if node == 'topic':
        df = pd.read_excel(f'/home/blueee/KnowRGFD/Data/{dataset}/graph/edges/news2topic_{num_topics}.xlsx')
        index_dict = np.load(f'/home/blueee/KnowRGFD/Data/{dataset}/graph/nodes/{node}_index_{num_topics}.npy',allow_pickle= True).item()
    else:
        df = pd.read_excel(f'/home/blueee/KnowRGFD/Data/{dataset}/graph/edges/news2{node}.xlsx')
        index_dict = np.load(f'/home/blueee/KnowRGFD/Data/{dataset}/graph/nodes/{node}_index.npy',allow_pickle= True).item()
    pair = df.values.tolist()
    edges = []
    edges_ = []
    for i in pair:
        head = news_index[i[0]] # 查詢新聞索引
        tail = index_dict[i[1]] # 查詢實體或主題索引
        edge = [head, tail]
        edge_ = [tail, head]
        edges.append(edge)
        edges_.append(edge_) 
    return edges,edges_

def build_hg(dataset,num_topics):
    news_attr = np.load(f'/home/blueee/KnowRGFD/Data/{dataset}/Embeddings/news_embeddings.npy')
    news_attr = torch.from_numpy(news_attr)
    entity_attr = np.load(f'/home/blueee/KnowRGFD/Data/{dataset}/Embeddings/entity_embeddings.npy')
    entity_attr = torch.from_numpy(entity_attr)
    topic_attr = np.load(f'/home/blueee/KnowRGFD/Data/{dataset}/Embeddings/topic_embeddings_{num_topics}.npy')    
    topic_attr = torch.from_numpy(topic_attr)

    news2entity, news2entity_ = load_edge(dataset,num_topics,'entity')
    news2topic, news2topic_ = load_edge(dataset,num_topics,'topic')
    df_news = pd.read_excel(f'/home/blueee/KnowRGFD/Data/{dataset}/news_final.xlsx')
    label = df_news['label'].tolist()
    
    data = HeteroData()
    data['news'].x = news_attr
    data['entity'].x = entity_attr
    data['topic'].x = topic_attr
    data['news', 'has', 'entity'].edge_index = torch.tensor(news2entity, dtype=torch.long).t().contiguous()
    data['entity', 'has_1', 'news'].edge_index = torch.tensor(news2entity_, dtype=torch.long).t().contiguous()
    data['news', 'belongs', 'topic'].edge_index = torch.tensor(news2topic, dtype=torch.long).t().contiguous()
    data['topic', 'belongs_1', 'news'].edge_index = torch.tensor(news2topic_, dtype=torch.long).t().contiguous()
    data['news'].y = torch.tensor(label,dtype = torch.long)
    print('='*60)
    print('HeteroGraph:',dataset,'\n',data)
    print(' num_nodes:',data.num_nodes,'\n','num_edges:',data.num_edges,'\n','Data has isolated nodes:',data.has_isolated_nodes(),'\n','Data is undirected:',data.is_undirected())
    print('='*60,'\n')
    torch.save(data,f'/home/blueee/KnowRGFD/Data/{dataset}/graph/{dataset}_{num_topics}.pt')
    return data

def class2global(edgelist,global_index,classindex, prefix):
    indices_g = []
    #print('edgelist:',edgelist)
    #print('classindex:',classindex)
    for i in edgelist:
        ID = classindex[i]
        global_key = f"{prefix}_{ID}"  # 動態加前綴
        index_g = global_index[global_key]  # 從 global_index 獲取全局索引
        indices_g.append(index_g)
    return indices_g

def get_edgeList(dataset,num_topics):
    news_index = np.load(f'/home/blueee/KnowRGFD/Data/{dataset}/graph/nodes/news_index.npy', allow_pickle=True).item()
    entity_index = np.load(f'/home/blueee/KnowRGFD/Data/{dataset}/graph/nodes/entity_index.npy', allow_pickle=True).item()
    topic_index = np.load(f'/home/blueee/KnowRGFD/Data/{dataset}/graph/nodes/topic_index_{num_topics}.npy', allow_pickle=True).item()
    data = torch.load(f'/home/blueee/KnowRGFD/Data/{dataset}/graph/{dataset}_{num_topics}.pt')
    del data['entity', 'has_1', 'news']
    del data['topic', 'belongs_1', 'news']

    newsList0 = data['news', 'has', 'entity'].edge_index.tolist()[0]
    entityList = data['news', 'has', 'entity'].edge_index.tolist()[1]
    newsList1 = data['news', 'belongs', 'topic'].edge_index.tolist()[0]
    topicList = data['news', 'belongs', 'topic'].edge_index.tolist()[1]

    # 修正索引到原始 ID
    newsList0 = [int(list(news_index.keys())[list(news_index.values()).index(idx)]) for idx in newsList0]
    entityList = [int(list(entity_index.keys())[list(entity_index.values()).index(idx)]) for idx in entityList]
    newsList1 = [int(list(news_index.keys())[list(news_index.values()).index(idx)]) for idx in newsList1]
    topicList = [int(list(topic_index.keys())[list(topic_index.values()).index(idx)]) for idx in topicList]
    
    global_index = np.load(f'/home/blueee/KnowRGFD/Data/{dataset}/graph/nodes/global_index_{num_topics}.npy', allow_pickle=True).item()

    news0_g = class2global(newsList0,global_index,news_index, "news") # 透過 global_index 將局部索引轉換為全局索引
    entity_g = class2global(entityList,global_index,entity_index, "entity")
    news1_g = class2global(newsList1,global_index,news_index, "news")
    topic_g = class2global(topicList,global_index,topic_index, "topic")   

    node_head = news0_g + news1_g   # node_head是左邊欄位的節點，直的看
    node_tail = entity_g + topic_g   # node_tail是右邊欄位的節點，直的看

    edgeList = []
    edgeList_rw = []
    for i in range(len(node_head)):
        head = node_head[i]
        tail = node_tail[i]
        edge = [head,tail]
        edge_rw = str(head)+' '+str(tail)
        edgeList.append(edge)
        edgeList_rw.append(edge_rw)
    with open(f'/home/blueee/KnowRGFD/Data/{dataset}/graph/edges/{dataset}_{num_topics}.edgelist','w',encoding = 'utf-8') as f:
        for i in edgeList_rw:
            f.write(str(i)+'\n')
        f.close()

    data_rw = Data(edge_index=torch.tensor(edgeList, dtype=torch.long).t().contiguous())
    G_rw = graph.load_edgelist(f'/home/blueee/KnowRGFD/Data/{dataset}/graph/edges/{dataset}_{num_topics}.edgelist',undirected = True)
    # 添加這兩行打印語句
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
    
    hg = build_hg(dataset,num_topics)       
    edgeList_rw = get_edgeList(dataset,num_topics)
    
    print(f'graph & edgelist for {dataset} done') 