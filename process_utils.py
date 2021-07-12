import os
import numpy as np
import jieba
import MeCab
import ipadic
import nltk

def str_insert(str_origin, pos, str_add):
    str_list = list(str_origin)
    str_list.insert(pos, str_add)
    str_out = ''.join(str_list)
    return str_out

def obtain_available_lps(corr_dir):
    ex_testset = ['newstestB2020', 'newstestM2020', 'newstestM2020', 'newstestP2020', 'newstestPa2020', 'newstestPb2020']
    corr_files = os.listdir(corr_dir)
    corr_files = [i for i in corr_files if i.endswith('.csv')]
    if corr_files[0].find('DA') == -1:
        avail_lps = [file.split('-')[-1].replace('.csv','') for file in corr_files]
    else:
        avail_lps = [file.split('-')[2] for file in corr_files if file.split('-')[1] not in ex_testset]
    avail_lps = [str_insert(i, 2, '-') for i in avail_lps]
    assert len(set(avail_lps)) == len(avail_lps)
    return avail_lps

def construct_file_name(lp_name, query_type, testset):
    src_name = lp_name.split('-')[0]
    tgt_name = lp_name.split('-')[-1]
    lp_name_mid = lp_name.replace('-', '')
    file_name = ''
    if query_type == 'src':
        file_name = testset + '-' + lp_name_mid + '-' + query_type + '.' + src_name
    elif query_type == 'ref':
        file_name = testset + '-' + lp_name_mid + '-' + query_type + '.' + tgt_name
    else:
        return NotImplementedError

    if testset != 'newstest2020' :
        return file_name
    else:
        return file_name + '.txt'

def seg_chinese(text):
    len_arr = []
    token_arr = []
    for i in text:
        seg_list = list(jieba.cut(i.strip()))
        seg_list = [i for i in seg_list if i != ' ']
        len_arr.append(len(seg_list))
        token_arr += seg_list
    return np.array(len_arr), token_arr

def seg_japanese(text):
    len_arr = []
    token_arr = []
    tagger = MeCab.Tagger(ipadic.MECAB_ARGS + " -Owakati")
    d = tagger.dictionary_info()
    assert d.size == 392126, "Please make sure to use the IPA dictionary for MeCab"
    for line in text:
        line = line.strip()
        sentence = tagger.parse(line).strip().split()
        sentence = [i for i in sentence if i != ' ']
        len_arr.append(len(sentence))
        token_arr += sentence
    return np.array(len_arr), token_arr

def compute_length_corpus(corpus_file):
    token_arr = []
    sents_len = []
    with open(corpus_file, 'r', encoding='utf8') as f:
        sents = f.readlines()
    if 'newstest2020' in corpus_file:
        lang = corpus_file.split('.')[-2]
    else:
        lang = corpus_file.split('.')[-1]
    if lang == 'zh':
        return seg_chinese(sents)
    elif lang == 'ja':
        return seg_japanese(sents)
    else:
        # seg_tokens = nltk.word_tokenize(i)
        for i in sents:
            seg_arr = nltk.word_tokenize(i)
            sents_len.append(len(seg_arr))
            token_arr += seg_arr
        return np.array(sents_len), token_arr

def stat_res_year():
    years = os.listdir('dumps/vat')
    stat_res = {}
    for year in years:
        stat_res[year] = {'num':0, 'token': 0, 'words':[]}
        files_path = os.path.join('dumps/vat', year)
        for file in os.listdir(files_path):
            len_arr, token_arr = compute_length_corpus(os.path.join(files_path, file))
            stat_res[year]['num'] += len(len_arr)
            stat_res[year]['token'] += sum(len_arr)
            stat_res[year]['words'] += token_arr
        stat_res[year]['words'] = len(set(stat_res[year]['words']))
    print(stat_res)

def stat_res_lp():
    years = os.listdir('dumps/vat')
    stat_res = {}
    for year in years:
        if year not in stat_res.keys():
            stat_res[year] = {}
        files_path = os.path.join('dumps/vat', year)
        for file in os.listdir(files_path):
            if '-src' in file:
                continue
            lp = file.split('-')[1]
            if lp not in stat_res[year].keys():
                stat_res[year][lp] = {'num':0, 'token': 0, 'words':[]}
            len_arr, token_arr = compute_length_corpus(os.path.join(files_path, file))
            stat_res[year][lp]['num'] += len(len_arr)
            stat_res[year][lp]['token'] += sum(len_arr)
            stat_res[year][lp]['words'] += token_arr

    for year in stat_res.keys():
        for lp in stat_res[year].keys():
            stat_res[year][lp]['words'] = len(set(stat_res[year][lp]['words']))
            res_str = year + '\t' + str_insert(lp,2,'-') + '\t' + str(stat_res[year][lp]['num']) + '\t' + str(stat_res[year][lp]['token']) + '\t' + str(stat_res[year][lp]['words'])
            print(res_str)
    print(stat_res)