# -*- coding: utf-8 -*-
import random
import codecs

PREFIX_THRESH = 3000

OUTPUT_FEATURES = ['id1', 'id2', 'answer', 'tag1', 'tag2',
                   'common_prefix_len', 'common_prefix_len_rel', # 5 6
                   'levenshtein', 'jaro_winkler',
                   'freq1', 'freq2', 'suf2freq1', 'suf2freq2', # 9 10 11 12
                   'suf3freq1', 'suf3freq2', 'suf4freq1', 'suf4freq2', # 13 14 15 16
                   'wv_dist', 'mut_info_corpus', 'mut_info_wiki', # 17 18 19
                   'freq_common_prefix', 'freq_suffix1', 'freq_suffix2', # 20 21 22
                   'jaro', # 23
                   #'log_freq_common_prefix', 'log_freq_suffix1', 'log_freq_suffix2', # 24 25 26
                   'w1', 'w2']

RU_ALPHABET = set(u'абвгдежзийклмнопрстуфхцчшщъыьэюяёabcdefghijklmnopqrstuvwxyz')
ES_ALPHABET = set(u'abcdefghijklmnopqrstuvwxyzñ')
HR_ALPHABET = set(u'abcdefghijklmnopqrstuvwxyzčćžđšž')


def gen_positive_examples(lemm_map):
  positive_examples = []
  for k, v in lemm_map.iteritems():
    if len(v) == 1 and len(v.values()[0]) == 1:
      continue
    for k1, v1 in v.items():
      for k2, v2 in v.items():
        if k1 == k2:
          if len(v1) > 1:
            positive_examples.append( tuple(random.sample(v1, 2)) )
        else:
          positive_examples.append( (random.choice(list(v1)), random.choice(list(v2))) )
  random.shuffle(positive_examples)
  return positive_examples

def save_features(filename, features):
  with codecs.open(filename, 'w', encoding='utf-8') as f:
    for feat in features:
      print >>f, ('\t'.join([unicode(feat[feat_name]) for feat_name in OUTPUT_FEATURES]))

def common_prefix_len(w1, w2):
  cpl = 0  #common prefix len
  while cpl < min(len(w1), len(w2)) and w1[cpl] == w2[cpl]:
    cpl += 1
  return cpl
