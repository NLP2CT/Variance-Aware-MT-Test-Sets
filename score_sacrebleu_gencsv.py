import os
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd

def read_text_score(file_path):
    scores = []
    with open(file_path, 'r', encoding='utf8') as f:
        scores = f.readlines()
    scores = [float(i.strip()) for i in scores if len(i) > 1]
    return scores

def test():
    sys1 = read_text_score('sents_score/wmt20/zh-en/bleu/newstest2020.zh-en.DeepMind.381.bleu')
    sys2 = read_text_score('sents_score/wmt20/zh-en/bleu/newstest2020.zh-en.Huoshan_Translate.919.bleu')
    sys2_chrf = read_text_score('sents_score/wmt20/zh-en/chrf/newstest2020.zh-en.Huoshan_Translate.919.chrf')
    assert len(sys1) == len(sys2) == len(sys2_chrf) == 2000

def load_data(data_dir, testset, **kwargs):
    info_table = pd.DataFrame({
        'TESTSET': [],
        'LP': [],
        'ID': [],
        'METRIC': [],
        'SYS': [],
        'SCORE': []
    })
    year_dir = os.path.join(data_dir, testset)
    lps_list = os.listdir(year_dir)
    veri_len = -1
    for lp in tqdm(lps_list):
        print('Current LP: ' + lp)
        metrics_dir = os.path.join(year_dir, lp)
        for metric in os.listdir(metrics_dir):
            for sys in os.listdir(os.path.join(metrics_dir, metric)):
                sys_file_path = os.path.join(metrics_dir, metric, sys)
                meta_info = sys.split('.')
                # SYS_NAME = '.'.join(meta_info[-3:-1])
                scores = read_text_score(sys_file_path)
                len_scored = len(scores)
                if veri_len == -1:
                    veri_len = len_scored
                assert len_scored == veri_len , 'Abnormal File' + sys_file_path
                if testset == 'wmt20':
                    LP = meta_info[1]
                    SYS_NAME = '.'.join(sys.split('.')[2:-1])
                else:
                    LP = meta_info[-1]
                    SYS_NAME = '.'.join(sys.split('.')[1:-1])

                tmp_df = pd.DataFrame({
                    'TESTSET': [meta_info[0]] * len_scored,
                    'LP': [LP] * len_scored,
                    'ID': list(range(0, len_scored)),
                    'METRIC': [metric] * len_scored,
                    'SYS': [SYS_NAME] * len_scored,
                    'SCORE': scores
                })
                info_table = info_table.append(tmp_df)
        veri_len = -1
    return info_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir',type=str, help='sacrebleu saved results', required=True)
    parser.add_argument('--test-set',type=str, help='sacrebleu saved testset', required=True)
    args = parser.parse_args()

    df = load_data(args.save_dir, args.test_set)
    df['ID'] = df['ID'].astype(int)
    df.reset_index(drop=True).to_csv(args.test_set + '.csv')
