#!/usr/local/bin/python
# -*- coding: utf-8 -*-

# Build features for spanish corpus

import codecs
import os
import wv_common
import random
import math
import argparse
import datetime

from collections import defaultdict
import Levenshtein as levenshtein

from freqtrie import FreqTrie
from build_features_common import *

def build_spanish(args, prefixes_gen=None):
  corpus_files = os.listdir(args.es_corpus)
  sentences = []
  wordcount = 0
  for filename in corpus_files:
    print 'Processing %s' % filename
    with codecs.open(os.path.join(args.es_corpus, filename), encoding='utf-8') as f:
      sent = []
      for line_no, line in enumerate(f):
        if line.strip() == '' or line.strip()[0] == '<' :
          sentences.append(sent)
          sent = []
        else:
          try:
            word, lemm, morph, num = line.strip().split(' ')
            if num != '0':  # 0 is for punctuation ant other trash
              sent.append( (word, lemm, morph) )
              wordcount += 1
          except Exception:
            print line_no
            print line
  print 'Loaded sentences: %d, wordcount = %d' % (len(sentences), wordcount)

  word_to_sents = defaultdict(set)
  words = []
  sent_sets = []

  for sent_no, s in enumerate(sentences):
    sent_set = set()
    for i, w in enumerate(s):
      words.append( (w[0], w[1], sent_no, len(words)) )
      sent_set.add(w[0].lower())
    for w in sent_set:
      word_to_sents[w].add(len(sent_sets))
    sent_sets.append(sent_set)


  goodwords = []
  word_used = defaultdict(int)
  for w in words:
    key = (w[0], w[1])
    if word_used[key] < 10:
      word_used[key] += 1
      goodwords.append(w)

  print 'Total goodwords: %d' % len(goodwords)

  prefix_map = defaultdict(list)
  for w in goodwords:
    prefix_map[w[0][:2].lower()].append(w)
  print 'Splitting prefix map'
  while True:
    splitted = False
    for k in prefix_map.keys():
      if len(prefix_map[k]) > PREFIX_THRESH:
        lst = prefix_map.pop(k)
        print 'splitting %d items for key %s' % (len(lst), k)
        splitted = True
        for w in lst:
          prefix_map[w[0][:(len(k)+1)].lower()].append(w)
        break
    if not splitted:
      break
  prefix_map_keys = prefix_map.keys()
  print 'Prefix spanish map splitted onto %d groups' % len(prefix_map)
  print 'Prefix map spanish max pairs: %d' % sum([len(v) ** 2 for k, v in prefix_map.iteritems()])


  print 'Loading word2vec data...'
  vectors = wv_common.load_text_vectors(args.es_text_vectors)

  articles = []  # wikipedia articles
  word_to_articles = defaultdict(set)
  word_to_id = {}
  id_to_word = {}
  word_to_freq = defaultdict(int)

  print 'Processing wiki data...'
  with codecs.open(args.spanish_wikidata, encoding='utf-8') as f:
    for line_no, line in enumerate(f):
      elems = line.strip().split()
      if elems[0] == 'WRD':
        article_words_ids = set()
        for word in elems[1:]:
          wl = word.lower()
          word_to_freq[wl] += 1
          if wl not in word_to_id:
            idx = len(word_to_id)
            word_to_id[wl] = idx
            id_to_word[idx] = wl
          idx = word_to_id[wl]
          article_words_ids.add(idx)
          word_to_articles[wl].add(len(articles))
        articles.append(article_words_ids)
      if line_no % 10000 == 0:
        print 'Processed %d lines of wikidata' % line_no
      if line_no > 100000:
        break

  print 'Total wiki vocab size: %d' % (len(word_to_freq))
  total_wiki_words = float(sum(word_to_freq.values()))

  alphabet = set()
  for w in word_to_freq.iterkeys():
    alphabet |= set(w)
  print 'Alphabet size: %d' % len(alphabet)

  trie_fwd = FreqTrie(alphabet)
  trie_inv = FreqTrie(alphabet)

  print 'Building tries'
  alphabet = set(ES_ALPHABET)
  trie_skipped = 0
  for wf_no, wf in enumerate(word_to_freq.iteritems()):
    if len(set(wf[0]) & alphabet) > 0:
      trie_fwd.add(wf[0], wf[1])
      trie_inv.add(wf[0][::-1], wf[1])
    else:
      trie_skipped += 1
    if wf_no % 10000 == 0:
      print 'Added to trie: %d words, nodecounts: %d %d skipped: %d' % (wf_no, trie_fwd.nodecount(), trie_inv.nodecount(), trie_skipped)

  print 'Building freqs'
  # 2,3,4-suffixes
  suffix_freqs = defaultdict(int)
  for w in goodwords:
    wl = w[0].lower()
    suffix_freqs[wl[-2:]] += 1
    suffix_freqs[wl[-3:]] += 1
    suffix_freqs[wl[-4:]] += 1

  word_freqs = defaultdict(int)
  for w in goodwords:
    word_freqs[w[0].lower()] += 1

  prc = 0
  tst = 1000000
  for i in xrange(tst):
    w1, w2 = random.sample(goodwords, 2)
    if w1[1]==w2[1]:
      prc += 1
  print 'prc:', prc, tst

  lemm_map = defaultdict(dict)
  for w in goodwords:
    lemm_map[w[1]][w[0]] = set()
  for i, w in enumerate(goodwords):
    lemm_map[w[1]][w[0]].add(i)  # lemm -> forms -> positions


  positive_examples = gen_positive_examples(lemm_map)
  print 'Generated %d positive examples' % len(positive_examples)

  print 'Generating features...'
  features = []
  if prefixes_gen == None:
    for i in xrange(args.gen_pos + args.gen_neg):
      if i != 0 and (i % 10000) == 0 :
        print '%d...' % i
      answer = True if i < args.gen_pos else False
      w1 = -1
      w2 = -1
      feature = None
      if answer:
        w1, w2 = positive_examples[i]
        feature = {'answer': answer, 'id1': goodwords[w1][3], 'id2': goodwords[w2][3]}
      else:
        prefix_group = None
        while True:
          prefix = random.choice(prefix_map_keys)
          prefix_group = prefix_map[prefix]
          w1 = random.randint(0, len(prefix_group)-1)
          w2 = random.randint(0, len(prefix_group)-1)
          t1 = words[prefix_group[w1][3]][0].lower()
          t2 = words[prefix_group[w2][3]][0].lower()
          if w1 == w2 or answer != (prefix_group[w1][1].lower() == prefix_group[w2][1].lower()) or t1 == t2:
            continue
          break
        feature = {'answer': answer, 'id1': prefix_group[w1][3], 'id2': prefix_group[w2][3]}
      features.append(feature)
  else:
    pass
    pos_ex = set([(min(wid1, wid2), max(wid1, wid2)) for wid1, wid2 in positive_examples])
    def fgen():
      for prefix_no, prefix in enumerate(sorted(prefix_map_keys)):
        print 'Processing prefix: %s (%d of %d) with %d words' % (prefix, prefix_no, len(prefix_map_keys), len(prefix_map[prefix]))
        prefix_group = []
        wcnt = defaultdict(int)
        for line in prefix_map[prefix]:
          if wcnt[line[0].lower()] < 1:
            wcnt[line[0].lower()] += 1
            prefix_group.append(line)
        print 'Lines in prefix group: %d' % (len(prefix_group))

        for i in xrange(len(prefix_group)):
          for j in xrange(i+1, len(prefix_group)):
            wid1 = prefix_group[i][3]
            wid2 = prefix_group[j][3]
            p_key = (min(wid1, wid2), max(wid1, wid2))
            feature = {'answer': p_key in pos_ex, 'id1': wid1, 'id2': wid2}
            yield feature
    features = fgen()


  print 'Filling features...'
  not_found_vectors_count = 0
  zero_mutual_info_corpus_count = 0
  zero_mutual_info_wiki_count = 0
  for feature_no, f in enumerate(features):
    i1 = f['id1'];  w1 = words[i1][0];  wl1 = w1.lower()
    i2 = f['id2'];  w2 = words[i2][0];  wl2 = w2.lower()
    f['tag1'] = '0'
    f['tag2'] = '0'
    f['w1'] = w1
    f['w2'] = w2
    # TODO: add tags: "max common length", "left-pos-tag" and "right-pos-tag" for both of them
    f['common_prefix_len'] = common_prefix_len(w1.lower(), w2.lower())
    f['common_prefix_len_rel'] = f['common_prefix_len'] * 2.0 / (len(w1) + len(w2))
    f['levenshtein'] = levenshtein.distance(wl1, wl2)
    f['jaro_winkler'] = levenshtein.jaro_winkler(wl1, wl2)
    f['jaro'] = levenshtein.jaro(wl1, wl2)

    common_prefix = wl1[:f['common_prefix_len']]
    suffix1 = wl1[f['common_prefix_len']:]
    suffix2 = wl2[f['common_prefix_len']:]

    f['freq_common_prefix'] = trie_fwd.get(common_prefix) * 1.0 / trie_fwd.getsum()
    f['freq_suffix1'] = trie_inv.get(suffix1) * 1.0 / trie_inv.getsum()
    f['freq_suffix2'] = trie_inv.get(suffix2) * 1.0 / trie_inv.getsum()

    f['freq1'] = word_freqs[w1.lower()]
    f['freq2'] = word_freqs[w2.lower()]

    f['suf2freq1'] = suffix_freqs[wl1[-2:]]
    f['suf2freq2'] = suffix_freqs[wl2[-2:]]
    f['suf3freq1'] = suffix_freqs[wl1[-3:]]
    f['suf3freq2'] = suffix_freqs[wl2[-3:]]
    f['suf4freq1'] = suffix_freqs[wl1[-4:]]
    f['suf4freq2'] = suffix_freqs[wl2[-4:]]

    if wl1 in vectors and wl2 in vectors:
      f['wv_dist']   = vectors[wl1]['vec'].dot(vectors[wl2]['vec'])
    else:
      not_found_vectors_count += 1
      f['wv_dist'] = 1.0

    # calculating mutual information
    w1fsc  = len(word_to_sents[wl1])
    w2fsc  = len(word_to_sents[wl2])
    w12fsc = len(word_to_sents[wl1] & word_to_sents[wl2])
    if w12fsc > 0:
      w1fsc /= 1.0 * len(sent_sets)
      w2fsc /= 1.0 * len(sent_sets)
      w12fsc /= 1.0 * len(sent_sets)
      f['mut_info_corpus'] = math.log(w1fsc) + math.log(w2fsc) - math.log(w12fsc)
    else:
      f['mut_info_corpus'] = 0.0
      zero_mutual_info_corpus_count += 1

    w1fsw  = len(word_to_articles[wl1])
    w2fsw  = len(word_to_articles[wl2])
    w12fsw = len(word_to_articles[wl1] & word_to_articles[wl2])
    if w12fsw > 0:
      w1fsw /= 1.0 * len(articles)
      w2fsw /= 1.0 * len(articles)
      w12fsw /= 1.0 * len(articles)
      f['mut_info_wiki'] = math.log(w1fsw) + math.log(w2fsw) - math.log(w12fsw)
    else:
      f['mut_info_wiki'] = 0.0
      zero_mutual_info_wiki_count += 1

    if feature_no % 1000 == 0:
      print 'Samples processed: %d' % (feature_no)

    if prefixes_gen != None:
      yield f

  if prefixes_gen == None:

    print 'Not found word vectors for:    %d pairs of %d' % (not_found_vectors_count, len(features))
    print 'Zeroed mutual corpus info for: %d pairs of %d' % (zero_mutual_info_corpus_count, len(features))
    print 'Zeroed mutual wiki info for:   %d pairs of %d' % (zero_mutual_info_wiki_count, len(features))

    print 'Saving features...'
    save_features(args.es_features_output, features)

def gen_spanish_corpus_morphology(path, output_filename):
  corpus_files = os.listdir(path)
  sentences = []
  wordcount = 0
  morphology = defaultdict(set)
  for filename_no, filename in enumerate(corpus_files):
    print 'Processing %s (%d of %d)' % (filename, filename_no, len(corpus_files))
    with codecs.open(os.path.join(path, filename), encoding='utf-8') as f:
      for line_no, line in enumerate(f):
        if line.strip() != '' and line.strip()[0] != '<' :
          try:
            word, lemm, morph, num = line.strip().split(' ')
            morphology[lemm.lower()].add(word.lower())
          except Exception:
            print line_no
            print line

  print 'Morphology keys: %d' % len(morphology)
  with codecs.open(output_filename, 'w', encoding='utf-8') as f:
    for k, words in sorted(morphology.iteritems()):
      for w in words:
        print >> f, w + '\t' + str(len(words)-1) + '\t' + '\t'.join([wf for wf in words if wf != w])


def main():
  start = datetime.datetime.now()
  random.seed()
  parser = argparse.ArgumentParser(description='Spanish corpus loader')
  parser.add_argument('--es-corpus', help='Spanish tagged corpus', default='../data/es/tagged_utf')
  parser.add_argument('--es-output-corpus-morphology', help='Output corpus morphology for spanish', default='../data/es/corpus_morphology.txt')
  args = parser.parse_args()

  gen_spanish_corpus_morphology(args.es_corpus, args.es_output_corpus_morphology)
  finish = datetime.datetime.now()
  print 'Time to run:', finish-start

if __name__=="__main__":
  main()
