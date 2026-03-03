# 合并训练集和测试集
# 按session id 对数据分块处理
# 先计算每个session id 的item_id的共现得分，再在全局按共现对分组求和，最后每个商品只保留得分最高的前N个共现商品
import pandas as pd
import glob
import os
import polars as pl
import logging
from argparse import ArgumentParser
from tqdm import tqdm
import sys 

# 将项目根目录（otto/）加入搜索路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 下面的 import 就能正常工作了
from utils.metrics import calculate_recall_at_20

sys.stdout.reconfigure(encoding='utf-8')


parser = ArgumentParser()
# 前导--表示可选参数，action='store_true'表示如果传入该参数则为True，否则为False
parser.add_argument('--submit', action='store_true', help='提交标志，传入即为true')
parser.add_argument('--version', required=True, help='版本号')
parser.add_argument('--eval', action='store_true', help='是否进行评估，传入即为true')
# 解析命令行参数，并去掉前导--，将参数存储在args对象中
args = parser.parse_args()

# 定义一个配置类，设置默认选项
class CFG:
    submit = args.submit
    train_path = '../inputs/train/train_valid/train_parquet'
    test_path = '../inputs/train/train_valid/test_parquet'
    lookback = 2
    topk = 50

if CFG.submit:
    CFG.train_path = '../inputs/submit/train_valid/train_parquet'
    CFG.test_path = '../inputs/submit/test/test_parquet'

agg_pkl_path = f'../inputs/comatrix/covisitation_{args.version}.pkl'
if not args.eval:
    if os.path.exists(agg_pkl_path):
        print(f"Loading existing co-visitation matrix from {agg_pkl_path}...")
        agg_df = pl.from_pandas(pd.read_pickle(agg_pkl_path))
    else:
        train_files = sorted(glob.glob(os.path.join(CFG.train_path, '*.parquet')))
        test_files = sorted(glob.glob(os.path.join(CFG.test_path, '*.parquet')))
        print(f'Loading {len(train_files)} train files and {len(test_files)} test files...')

        # 优化内存读取，仅保留需要的列
        def read_data(files):
            return pl.concat([pl.read_parquet(f).select(['session', 'aid', 'type']) for f in files], how='vertical')

        train_df = read_data(train_files)
        test_df = read_data(test_files)
        data_df = pl.concat([train_df, test_df], how='vertical')
        print(f'Total data shape: {data_df.shape}')

        weights = {
            0: 1.0, # clicks
            1: 6.0, # carts
            2: 3.0  # orders
        }
        if args.version == 'v2':
            weights = {0: 1.0, 1: 3.0, 2: 6.0}
        if args.version == 'v3':
            weights = {0: 1.0, 1: 2.0, 2: 4.0}

        # Polars 映射优化：type 转换为整数提高处理速度
        type_map = {'clicks': 0, 'carts': 1, 'orders': 2}
        data_df = data_df.with_columns([
            pl.col('type').replace(type_map).cast(pl.Int8).alias('type_int')
        ])
        data_df = data_df.with_columns([
            pl.col('type_int').replace(weights).alias('weight')
        ])

        # 按 session 分组而不是简单 apply，Polars 性能更强
        # 使用 session // 100000 增大分块，减少循环开销
        data_df = data_df.with_columns([
            (pl.col('session') // 100000).alias('chunk_id')
        ])

        def count(chunk):
            # 转换为 dict 累加比生成巨大 list 再转 df 快得多
            covisitation_dict = {}
            
            # 按 session 分组处理
            for session, df in chunk.group_by('session'):
                aids = df['aid'].to_list()
                weights = df['weight'].to_list()
                
                for i in range(len(aids)):
                    for j in range(max(0, i - CFG.lookback), i):
                        a = aids[j]
                        b = aids[i]
                        w = weights[i]
                        
                        # 双向共现（可选，根据需求）
                        pair = (a, b)
                        covisitation_dict[pair] = covisitation_dict.get(pair, 0) + w
                        
                        # 如果是对称共现，开启下面这行
                        # pair_rev = (b, a)
                        # covisitation_dict[pair_rev] = covisitation_dict.get(pair_rev, 0) + w

            # 转为 Polars DataFrame
            if not covisitation_dict:
                return None
                
            hist_aids = []
            future_aids = []
            scores = []
            for (a, b), w in covisitation_dict.items():
                hist_aids.append(a)
                future_aids.append(b)
                scores.append(w)
                
            return pl.DataFrame({
                'hist_aid': hist_aids,
                'aid_future': future_aids,
                'score': scores
            })

        aggs = []
        for chunk_id, chunk in tqdm(data_df.group_by('chunk_id'), desc='Processing chunks'):
            if len(chunk) < 2:
                continue
            agg = count(chunk)
            if agg is not None:
                aggs.append(agg)

        print("Merging chunks...")
        cnt_df = pl.concat(aggs, how='vertical')
        cnt_df = cnt_df.group_by(['hist_aid', 'aid_future']).agg(pl.sum('score').alias('score'))

        print("Generating top-k...")
        # 优化 Top-K 逻辑
        agg_df = cnt_df.sort(['hist_aid', 'score'], descending=[False, True]).group_by('hist_aid').head(CFG.topk)

        # 转换为列表形式以便保存 hist_aid, [aid_future1, aid_future2, ...], [score1, score2, ...]
        agg_df = agg_df.group_by('hist_aid').agg([
            pl.col('aid_future').alias('aid_future'),
            pl.col('score').alias('score')
        ])

        print(f'Final agg_df shape: {agg_df.shape}')

        if not os.path.exists('../inputs/comatrix'):
            os.makedirs('../inputs/comatrix')

        # 使用 polars 推荐的 parquet 格式或转 pandas 保存 pkl
        agg_pkl_path = f'../inputs/comatrix/covisitation_{args.version}.pkl'
        agg_df.to_pandas().to_pickle(agg_pkl_path)


if args.eval:
    if not os.path.exists(agg_pkl_path):
        raise FileNotFoundError(f"Co-visitation matrix not found at {agg_pkl_path}. Please run without --eval to generate it first.")
    agg_df = pl.from_pandas(pd.read_pickle(agg_pkl_path))

    # 召回验证集商品，对于每个历史 aid，获取 top-k 共现商品，并计算覆盖率
    print("Starting evaluation...")

    # 1. 加载验证集真实标签
    labels_path = '../inputs/train/train_valid/test_labels.parquet'
    labels_df = pl.read_parquet(labels_path)

    test_files = sorted(glob.glob(os.path.join(CFG.test_path, '*.parquet')))
    test_df = pl.concat(
        [pl.read_parquet(f).select(['session', 'aid', 'ts']) for f in test_files],
        how='vertical'
    )
    test_df = test_df.sort(['session','ts'])  

    # 2. 提取用户历史种子 (Seed Item)
    # 我们取每个 session 交互过的最后一个 aid 作为召回种子
    valid_test_history = test_df.group_by('session').tail(1) 

    # 3. 快速查找字典
    topk_lookup = {row['hist_aid']: row['aid_future'] for row in agg_df.to_dicts()}

    # 4. 生成预测 (保持原逻辑，取 Top 20)
    def get_candidates(row):
        hist_aid = row['aid']
        # 如果该 aid 没有任何共现，返回空列表
        return topk_lookup.get(hist_aid, [])[:20]

    valid_test_history = valid_test_history.with_columns([
        pl.struct(['aid']).map_elements(get_candidates, return_dtype=pl.List(pl.Int64)).alias('labels')
    ])

    # 5. 调用评估
    # 此时 labels_df 包含多行(clicks/carts/orders)，preds_df 是一行一个 session
    # calculate_recall_at_20 会在每一行标签上计算你的召回效果
    recall_score = calculate_recall_at_20(valid_test_history.select(['session', 'labels']), labels_df)
    print(f'Overall Recall@20: {recall_score:.4f}')