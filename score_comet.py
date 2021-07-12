import os
import pandas as pd
import numpy as np
import pathlib
import argparse
from tqdm import tqdm
from comet.models import download_model
from process_utils import construct_file_name, obtain_available_lps

model = download_model("wmt-large-da-estimator-1719")


def eval_single_sys(source, hypothesis, reference):
    data = {"src": source, "mt": hypothesis, "ref": reference}
    data = [dict(zip(data, t)) for t in zip(*data.values())]
    info_dict, scores = model.predict(data, cuda=True, show_progress=True)
    return info_dict, scores


def run_eval_batch(lps_source_dir, lps_hypo_dir, lps_ref_dir, lps_list, testset, debug=False):
    info_table = pd.DataFrame({
        'TESTSET': [],
        'LP': [],
        'ID': [],
        'METRIC': [],
        'SYS': [],
        'SCORE': []
    })

    for lp in tqdm(lps_list):
        print('Current LP: ' + lp)
        with open(os.path.join(lps_source_dir, construct_file_name(lp, 'src', testset)), 'r', encoding='utf8') as f:
            lp_source = f.readlines()
        with open(os.path.join(lps_ref_dir, construct_file_name(lp, 'ref', testset)), 'r', encoding='utf8') as f:
            lp_ref = f.readlines()
        lp_source = [i.rstrip() for i in lp_source]
        lp_ref = [i.rstrip() for i in lp_ref]
        hypo_dir = os.path.join(lps_hypo_dir, lp)
        cand_list = [os.path.join(hypo_dir, i) for i in os.listdir(hypo_dir) if i.lower().find('human') == -1]
        veri_len = -1  # Make sure the amount of sentences is the same for each system
        for cand in cand_list:
            with open(cand, 'r', encoding='utf8') as f:
                cand_hypo = f.readlines()
            cand_hypo = [i.rstrip() for i in cand_hypo]
            # ipdb.set_trace()
            assert len(cand_hypo) == len(lp_source) == len(lp_ref)
            info_dict, scores = eval_single_sys(source=lp_source, hypothesis=cand_hypo, reference=lp_ref)
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
                'METRIC': ['COMET'] * len_scored,
                'SYS': [SYS_NAME] * len_scored,
                'SCORE': scores
            })
            info_table = info_table.append(tmp_df)
        veri_len = -1
    return info_table

def parse_args():
    parser = argparse.ArgumentParser("Calculate COMET")
    parser.add_argument("--src-dir", default=None, type=str, help="path of WMT source files", required=True)
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
    lps_source_dir = args.src_dir
    avail_lps = obtain_available_lps(args.scores_dir)
    print(avail_lps)
    eval_res = run_eval_batch(lps_source_dir=lps_source_dir, lps_hypo_dir=lps_hypo_dir, lps_ref_dir=lps_ref_dir, lps_list=avail_lps, testset=args.testset_name)
    eval_res['ID'] = eval_res['ID'].astype(int)
    eval_res.to_csv(args.score_dump)

if __name__ == '__main__':
    main()