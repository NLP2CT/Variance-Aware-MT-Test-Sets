import os
import pandas as pd
import numpy as np
import sacrebleu
import pathlib
import argparse
from tqdm import tqdm
from process_utils import obtain_available_lps

def eval_sacrebleu_porter(metrics, test_set, tokenize, lp, cand_str, save_dir):
    for metric in metrics:
        for cand_sys in cand_str:
            test_set = 'wmt' + test_set[-2:]
            sacre_cmd = f'sacrebleu -t {test_set} -l {lp} --metrics {metric} --tokenize {tokenize} --score-only --sentence-level --input {cand_sys} --width 4'
            lp_res_dir = os.path.join(save_dir, test_set, lp, metric)
            # create dir if not exist
            if not os.path.exists(lp_res_dir):
                pathlib.Path(lp_res_dir).mkdir(parents=True, exist_ok=True)
            sacre_cmd += ' > ' + os.path.join(lp_res_dir, cand_sys.split('/')[-1].replace('.txt','.' + metric))
            # print(sacre_cmd)
            res = os.popen(sacre_cmd)
            output_str = res.read()
            # print(output_str)
            del output_str

def run_eval_batch(lps_hypo_dir, lps_list, **kwargs):

    defualt_tok = kwargs['tokenize']
    for lp in tqdm(lps_list):
        print('Current LP: ' + lp)
        hypo_dir = os.path.join(lps_hypo_dir, lp)
        cand_str = [os.path.join(lps_hypo_dir,lp ,i) for i in os.listdir(hypo_dir) if i.lower().find('human') == -1]
        tgt_lang = lp.split('-')[-1]
        kwargs['lp'] = lp
        kwargs['cand_str'] = cand_str
        # handle ja and zh segmentation
        if tgt_lang == 'zh':
            kwargs['tokenize'] = 'zh'
        elif tgt_lang == 'ja':
            kwargs['tokenize'] = 'ja-mecab'
        else:
            kwargs['tokenize'] = defualt_tok
        eval_sacrebleu_porter(**kwargs)

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--topk',default=False, action="store_true")
    parser.add_argument('--metrics', '-m', choices=sacrebleu.metrics.METRICS.keys(), nargs='+', default=None, help='metrics to compute (default: bleu)', required=True)
    parser.add_argument('--test-set',type=str, help='save eval results directory', required=True)
    parser.add_argument('--tokenize', '-tok', choices=sacrebleu.TOKENIZERS.keys(), default='intl', help='Tokenization method to use for BLEU. If not provided, defaults to `zh` for Chinese, `mecab` for Japanese and `mteval-v13a` otherwise.', required=True)

    # VAT arguments
    parser.add_argument("--hypos-dir", default=None, type=str, help="path of WMT system hypos", required=True)
    parser.add_argument('--save-path',type=str, help='save eval results directory', required=True)
    parser.add_argument("--scores-dir", default=None, type=str, help="path of WMT DA files", required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    lps_hypo_dir = args.hypos_dir
    avail_lps = obtain_available_lps(args.scores_dir)
    run_eval_batch(lps_hypo_dir=lps_hypo_dir, lps_list=avail_lps, save_dir=args.save_path, metrics=args.metrics, test_set=args.test_set, tokenize=args.tokenize)


if __name__ == '__main__':
    '''
    python score_sacrebleu.py --metric bleu chrf --test-set wmt2020 --save-path sents_score --tokenize intl --hypos-dir ${WMT_DATA_PATH}/system-outputs --scores-dir ${WMT_DATA_PATH}/results/system-level/scores_ALL
    '''
    main()