import os
import pathlib
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_text_score(file_path):
    scores = []
    with open(file_path, 'r', encoding='utf8') as f:
        scores = f.readlines()
    scores = [float(i.strip()) for i in scores if len(i) > 1]
    return scores

def read_dump_csv(dump_file):
    score_df = pd.read_csv(dump_file)
    score_df = score_df.drop(['Unnamed: 0'],axis=1)
    return score_df

def cal_diff_measure(score_df, measure):
    if measure == 'mean':
        diff_mean = score_df.groupby('ID')['SCORE'].mean()
    elif measure == 'std':
        diff_mean = score_df.groupby('ID')['SCORE'].std()
    else:
        raise "Unsupported measurement!"
    # ensure an sequence ordered by id
    id2diff_list = list(zip(diff_mean.index.to_list(), diff_mean.to_list()))
    id2diff = zip(diff_mean.index.to_list(), diff_mean.to_list())
    ordered_diff = [i[1] for i in sorted(id2diff_list, key=lambda x:x[0])]
    assert len(ordered_diff) == len(diff_mean)
    return ordered_diff, id2diff

def diff_to_file(out_path, metric, diff_array, zip_instance, dump_json=False):
    with open(os.path.join(out_path, metric + '.pkl'),'wb') as f:
        pickle.dump(diff_array, f)
    if dump_json:
        diff_dict = dict(zip_instance)
        with open(os.path.join(out_path, metric + '.json'),'w', encoding='utf8') as f:
            json.dump(diff_dict,f,sort_keys=True)

def dump_filter_to_json(filter_dict, output_dir, measure, filter_per):
    metrics = filter_dict.keys()
    for metric in metrics:
        if not os.path.exists(output_dir):
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(output_dir, metric+'_filter-' + measure + str(filter_per)+'.json')
        with open(out_path, 'w', encoding='utf8') as f:
            json.dump(filter_dict[metric], f)

def main(dump_file, output_dir, measure, filter_per):
    scores_df = read_dump_csv(dump_file)
    lps_list = set(scores_df['LP'].values)
    metrics = set(scores_df['METRIC'].values)
    metrics_res = dict.fromkeys(metrics)
    for metric in metrics:
        metrics_res[metric] = dict.fromkeys(lps_list)
    print("Language Pairs: ", lps_list)
    for lp in tqdm(lps_list):
        LP_df = scores_df[(scores_df.LP == lp)]
        for metric in metrics:
            sub_df = LP_df[(LP_df.METRIC == metric)]
            diff_array, id2diff = cal_diff_measure(sub_df,measure=measure)
            diff_array = np.array(diff_array)
            threshold = np.percentile(diff_array, filter_per)
            sampled_indices = np.where(diff_array > threshold)[0]
            metrics_res[metric][lp] = sampled_indices.tolist()
    dump_filter_to_json(metrics_res, output_dir, measure, filter_per)


if __name__ == "__main__":
    '''
    python cal_filtering.py --score-dump wmt20-bertscore.csv --output VAT_meta/wmt20-test/ --filter-per 60
    '''
    parser = argparse.ArgumentParser("Construct VAT meta-information (reserved indices of the original test set)")
    parser.add_argument("--score-dump", default=None, help="CSV file that strores the sentence-level scores of various systems", required=True)
    parser.add_argument("--output", default=None, help="Directory for saving the meta-information", required=True)
    parser.add_argument("--measurement", choices={'std','mean'}, default='std', help="Filtering statistics. Default: Std. as variance")
    parser.add_argument("--filter-per", type=int, default=None, help="Filtering percentage", required=True)

    args = parser.parse_args()
    main(dump_file=args.score_dump, output_dir=args.output, measure=args.measurement, filter_per=args.filter_per)