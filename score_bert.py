import os
import argparse
import torch

import gc
import bert_score
import pandas as pd
from tqdm import tqdm
from process_utils import obtain_available_lps, construct_file_name

def gen_score_table(cand, scores, metric_type, testset):
    len_scored = len(scores)
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
        'METRIC': [metric_type] * len_scored,
        'SYS': [SYS_NAME] * len_scored,
        'SCORE': scores
    })
    return tmp_df

def bert_score_main(args):
    torch.multiprocessing.set_sharing_strategy("file_system")

    if os.path.isfile(args.cand):
        with open(args.cand) as f:
            cands = [line.strip() for line in f]

        refs = []
        for ref_file in args.ref:
            assert os.path.exists(ref_file), f"reference file {ref_file} doesn't exist"
            with open(ref_file) as f:
                curr_ref = [line.strip() for line in f]
                assert len(curr_ref) == len(cands), f"# of sentences in {ref_file} doesn't match the # of candidates"
                refs.append(curr_ref)
        refs = list(zip(*refs))
    elif os.path.isfile(args.ref[0]):
        assert os.path.exists(args.cand), f"candidate file {args.cand} doesn't exist"
    else:
        cands = [args.cand]
        refs = [args.ref]
        assert not args.idf, "do not support idf mode for a single pair of sentences"

    all_preds, hash_code = bert_score.score(
        cands,
        refs,
        model_type=args.model,
        num_layers=args.num_layers,
        verbose=args.verbose,
        idf=args.idf,
        batch_size=args.batch_size,
        lang=args.lang,
        return_hash=True,
        rescale_with_baseline=args.rescale_with_baseline,
        baseline_path=args.baseline_path,
    )
    avg_scores = [s.mean(dim=0) for s in all_preds]
    P = avg_scores[0].cpu().item()
    R = avg_scores[1].cpu().item()
    F1 = avg_scores[2].cpu().item()
    msg = hash_code + f" P: {P:.6f} R: {R:.6f} F1: {F1:.6f}"
    print(msg)
    if args.seg_level:
        ps, rs, fs = all_preds
        return ps, rs, fs

def eval_main(args, lps_hypo_dir, lps_ref_dir, lps_list, testset):

    info_table = pd.DataFrame({
        'TESTSET': [],
        'LP': [],
        'ID': [],
        'METRIC': [],
        'SYS': [],
        'SCORE': []
    })
    if args.debug is True:
        lps_list = lps_list[:1]
        print(lps_list)

    for lp in tqdm(lps_list):
        print('Current LP: ' + lp)
        hypo_dir = os.path.join(lps_hypo_dir, lp)
        cand_list = [os.path.join(hypo_dir, i) for i in os.listdir(hypo_dir) if i.lower().find('human') == -1]
        tgt_lang = lp.split('-')[-1]
        lp_ref = [os.path.join(lps_ref_dir, construct_file_name(lp, 'ref', testset))]
        veri_len = -1  # Make sure the amount of sentences is the same for each system
        args.lang = tgt_lang

        if args.debug:
            cand_list = cand_list[:1]

        for cand in cand_list:
            args.ref = lp_ref
            args.cand = cand
            ps, rs, fs = bert_score_main(args)
            len_scored = len(ps)
            if veri_len == -1:
                veri_len = len_scored
            assert len_scored == veri_len == len(rs) == len(fs), 'Abnormal File' + cand
            bert_score_res = {
                'bert-p': ps,
                'bert-r': rs,
                'bert-f': fs,
            }
            for sub_metric in bert_score_res.keys():
                tmp_df = gen_score_table(cand=cand, scores=bert_score_res[sub_metric], metric_type=sub_metric, testset=testset)
                info_table = info_table.append(tmp_df)
        veri_len = -1
        gc.collect()
        torch.cuda.empty_cache()  # free gpu memory, avoid oom
    return info_table


def parse_args():
    parser = argparse.ArgumentParser("Calculate BERTScore")

    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help='two-letter abbreviation of the language (e.g., en) or "en-sci" for scientific text',
    )
    parser.add_argument(
        "-m", "--model", default=None, help="BERT model name (default: bert-base-uncased) or path to a pretrain model"
    )
    parser.add_argument("-l", "--num_layers", type=int, default=None, help="use first N layer in BERT (default: 8)")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size (default: 64)")
    parser.add_argument("--nthreads", type=int, default=4, help="number of cpu workers (default: 4)")
    parser.add_argument("--idf", action="store_true", help="BERT Score with IDF scaling")
    parser.add_argument(
        "--rescale_with_baseline", action="store_true", help="Rescaling the numerical score with precomputed baselines"
    )
    parser.add_argument("--baseline_path", default=None, type=str, help="path of custom baseline csv file")
    parser.add_argument("-s", "--seg_level", action="store_true", help="show individual score of each pair")
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("-r", "--ref", type=str, nargs="+", required=True, help="reference file path(s) or a string")
    parser.add_argument(
        "-c", "--cand", type=str, required=True, help="candidate (system outputs) file path or a string"
    )
    parser.add_argument("-d", "--debug", action="store_true", help="debug switch")

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
    eval_res = eval_main(args=args, lps_hypo_dir=lps_hypo_dir, lps_ref_dir=lps_ref_dir, lps_list=avail_lps, testset=args.testset_name)
    eval_res['ID'] = eval_res['ID'].astype(int)
    eval_res.to_csv(args.score_dump)


if __name__ == "__main__":
    main()