import sys
import os
import gc
import torch
import argparse
from tqdm import tqdm
from datetime import datetime
from process_utils import construct_file_name, obtain_available_lps

import pandas as pd

from bleurt import score

startTime = datetime.now()


def run_bleurt(candidates: list,
               references: list,
               scorer):
    scores = scorer.score(references=references, candidates=candidates)
    return scores


def run_eval_batch(checkpoint, lps_hypo_dir, lps_ref_dir, lps_list, testset):
    info_table = pd.DataFrame({
        'TESTSET': [],
        'LP': [],
        'ID': [],
        'METRIC': [],
        'SYS': [],
        'SCORE': []
    })

    scorer = score.BleurtScorer(checkpoint)

    for lp in tqdm(lps_list):
        print('Current LP: ' + lp)
        lp_startTime = datetime.now()
        # with open(os.path.join(lps_source_dir, construct_file_name(lp, 'src')), 'r', encoding='utf8') as f:
        #     lp_source = f.readlines()
        with open(os.path.join(lps_ref_dir, construct_file_name(lp, 'ref', testset)), 'r', encoding='utf8') as f:
            lp_ref = f.readlines()
        # lp_source = [i.rstrip() for i in lp_source]
        lp_ref = [i.rstrip() for i in lp_ref]
        hypo_dir = os.path.join(lps_hypo_dir, lp)
        cand_list = [os.path.join(hypo_dir, i) for i in os.listdir(hypo_dir) if i.lower().find('human') == -1]
        veri_len = -1  # Make sure the amount of sentences is the same for each system
        for cand in cand_list:
            with open(cand, 'r', encoding='utf8') as f:
                cand_hypo = f.readlines()
            cand_hypo = [i.rstrip() for i in cand_hypo]
            assert len(cand_hypo) == len(lp_ref)
            scores = run_bleurt(candidates=cand_hypo, references=lp_ref, scorer=scorer)
            # Merge into the scoring table
            len_scored = len(scores)
            if veri_len == -1:
                veri_len = len_scored
            assert len_scored == veri_len, 'Abnormal File' + cand
            meta_info = cand.split('/')[-1].split('.')
            # SYS_NAME = '.'.join(meta_info[-3:-1])

            if testset == 'newstest2020':
                LP = meta_info[1]
                SYS_NAME = '.'.join(cand.split('.')[2:-1])
            else:
                LP = meta_info[-1]
                SYS_NAME = '.'.join(cand.split('.')[1:-1])

            tmp_df = pd.DataFrame({
                'TESTSET': [meta_info[0]] * len_scored,
                'LP': [LP] * len_scored,
                'ID': list(range(0, len_scored)),
                'METRIC': ['BLEURT'] * len_scored,
                'SYS': [SYS_NAME] * len_scored,
                'SCORE': scores
            })
            info_table = info_table.append(tmp_df)
            gc.collect()
            torch.cuda.empty_cache()
        veri_len = -1
        print("Runtime: {}\n".format(datetime.now() - lp_startTime))
    print("Total Runtime: {}".format(datetime.now() - startTime))
    return info_table

def parse_args():
    parser = argparse.ArgumentParser("Calculate BLEURT")
    parser.add_argument("--checkpoint", default=None, type=str, help="path of BLEURT checkpoint", required=True)
    # VAT arguments
    parser.add_argument("--hypos-dir", default=None, type=str, help="path of WMT system hypos", required=True)
    parser.add_argument("--refs-dir", default=None, type=str, help="path of WMT references", required=True)
    parser.add_argument("--scores-dir", default=None, type=str, help="path of WMT DA files", required=True)
    parser.add_argument("--testset-name", default=None, type=str, help="name of the testset", required=True)
    parser.add_argument("--score-dump", default=None, type=str, help="name of the saved scoring CSV file", required=True)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    lps_hypo_dir = args.hypos_dir
    lps_ref_dir = args.refs_dir
    avail_lps = obtain_available_lps(args.scores_dir)
    print(avail_lps)
    eval_res = run_eval_batch(checkpoint=args.checkpoint, lps_hypo_dir=lps_hypo_dir, lps_ref_dir=lps_ref_dir, lps_list=avail_lps, testset=args.testset_name)
    eval_res['ID'] = eval_res['ID'].astype(int)
    eval_res.to_csv(args.score_dump)

if __name__ == "__main__":
    '''
    export WMT_DATA_PATH=/home/zhanrunzhe/workspace/diffculty-eval-nmt/test_data/wmt2020
    export BLEURT_CHECKPOINT=/home/zhanrunzhe/workspace/bleurt/bleurt-base-128
    CUDA_VISIBLE_DEVICES=1 python score_bleurt --checkpoint ${BLEURT_CHECKPOINT} --hypos-dir ${WMT_DATA_PATH}/system-outputs --refs-dir ${WMT_DATA_PATH}/references --scores-dir ${WMT_DATA_PATH}/results/system-level/scores_ALL --testset-name newstest2020 --score-dump wmt20-bleurt.csv
    '''
    main()