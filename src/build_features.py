#!/usr/local/bin/python
# -*- coding: utf-8 -*-

# Build features for croatian and nuts corpuses

import argparse
import codecs
import random
import cPickle
import wv_common
import math
import datetime

import Levenshtein as levenshtein
from collections import defaultdict

from freqtrie import FreqTrie
from build_features_common import *
import spanish

'''
Берем:
A : 19783, unique words: 7434, unique lemms: 2868  - Прилагательные
N : 59789, unique words: 14019, unique lemms: 8721 - Существительные
R : 5888, unique words: 899, unique lemms: 729     - Наречия
V : 25925, unique words: 4220, unique lemms: 1728  - Глаголы

Не берем
C : 9550, unique words: 48, unique lemms: 32   - Всякие союзы и местоимения
M : 4343, unique words: 704, unique lemms: 646 - Числительные цифрами и буквами, иногда даже склоняются
Q : 740, unique words: 17, unique lemms: 13    - Частицы с местоимениями
P : 9009, unique words: 329, unique lemms: 53  - Указательные и личные местоимения, много склюняются
S : 17429, unique words: 92, unique lemms: 57  - Предлоги
Y : 5, unique words: 5, unique lemms: 5        - Аббревиатуры и сокращения
X : 1198, unique words: 263, unique lemms: 294 - Имена собственные, в т.ч. и на других языках
Z : 25322, unique words: 42, unique lemms: 41  - Пунктуация

Априорная вероятность того, что два слова из одной леммы - 0.001

'''


def build_nuts(args, prefixes_gen=None):
  print "Loading nuts corpus"
  with open(args.nuts_corpus) as f:
    corpus = cPickle.load(f)

  print "Flattening nuts corpus"
  # addressation example: c['texts']['push']['paragraphs'][3][1][5] -> {lex, text, gr}
  sentences = []
  words = []
  for name, text in corpus['texts'].iteritems():
    for par in text['paragraphs']:
      for sent in par:
        sentences.append(sent)
  sent_sets = [] # For mutual information
  word_to_sents = defaultdict(set)
  # Here we have ~89K sentences with ~962K words
  for s in sentences:
    sent_set = set()
    for i, w in enumerate(s):
      w['lex']  = ''.join([ch for ch in w['lex']  if ch != '`'])
      w['text'] = ''.join([ch for ch in w['text'] if ch != '`'])
      gr = w['gr']
      w['id'] = i + 1
      if gr.startswith('S,'):
        w['tag'] = 'N'
      elif gr == 'ADV':
        w['tag'] = 'R'
      elif gr.startswith('A='):
        w['tag'] = 'A'
      elif gr.startswith('V,'):
        w['tag'] = 'V'
      else:
        w['tag'] = '0'
      words.append(w)
      if w['tag'] != '0':
        sent_set.add(w['text'].lower())
    for w in sent_set:
      word_to_sents[w].add(len(sent_sets))
    sent_sets.append(sent_set)


  tagstats = defaultdict(int)
  for i, w in enumerate(words):
    tagstats[w['tag']]+=1
    w['position'] = i
  for t, c in tagstats.iteritems():
    print t, c
  goodwords = [w for w in words if w['tag'] != '0']

  prefix_map = defaultdict(list)
  for w in goodwords:
    prefix_map[w['text'][:2].lower()].append(w)
  print 'Splitting prefix map'
  while True:
    splitted = False
    for k in prefix_map.keys():
      if len(prefix_map[k]) > PREFIX_THRESH:
        lst = prefix_map.pop(k)
        print 'splitting %d items for key %s' % (len(lst), k)
        splitted = True
        for w in lst:
          prefix_map[w['text'][:(len(k)+1)].lower()].append(w)
        break
    if not splitted:
      break
  prefix_map_keys = prefix_map.keys()
  print 'Prefix russian map splitted onto %d groups' % len(prefix_map)
  print 'Prefix map russian max pairs: %d' % sum([len(v) ** 2 for k, v in prefix_map.iteritems()])

  print 'Loading word2vec data...'
  vectors = wv_common.load_text_vectors(args.nuts_text_vectors)

  articles = []  # wikipedia articles
  word_to_articles = defaultdict(set)
  word_to_id = {}
  id_to_word = {}
  word_to_freq = defaultdict(int)

  print 'Processing wiki data...'
  with codecs.open(args.russian_wikidata, encoding='utf-8') as f:
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
      if line_no % 100000 == 0:
        print 'Processed %d lines of wikidata' % line_no

  print 'Total wiki vocab size: %d' % (len(word_to_freq))
  total_wiki_words = float(sum(word_to_freq.values()))

  alphabet = set()
  for w in word_to_freq.iterkeys():
    alphabet |= set(w)
  print 'Alphabet size: %d' % len(alphabet)

  trie_fwd = FreqTrie(alphabet)
  trie_inv = FreqTrie(alphabet)

  print 'Building tries'
  alphabet = set(RU_ALPHABET)
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
    wl = w['text'].lower()
    suffix_freqs[wl[-2:]] += 1
    suffix_freqs[wl[-3:]] += 1
    suffix_freqs[wl[-4:]] += 1

  word_freqs = defaultdict(int)
  for w in goodwords:
    word_freqs[w['text'].lower()] += 1

  prc = 0
  tst = 1000000
  for i in xrange(tst):
    w1, w2 = random.sample(goodwords, 2)
    if w1['lex']==w2['lex']:
      prc += 1
  print 'prc:', prc, tst

  lemm_map = defaultdict(dict)
  for w in goodwords:
    lemm_map[w['lex']][w['text']] = set()
  for i, w in enumerate(goodwords):
    lemm_map[w['lex']][w['text']].add(i)  # lemm -> forms -> positions


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
        feature = {'answer': answer, 'id1': goodwords[w1]['position'], 'id2': goodwords[w2]['position']}
      else:
        prefix_group = None
        while True:
          prefix = random.choice(prefix_map_keys)
          prefix_group = prefix_map[prefix]
          w1 = random.randint(0, len(prefix_group)-1)
          w2 = random.randint(0, len(prefix_group)-1)
          t1 = words[prefix_group[w1]['position']]['text'].lower()
          t2 = words[prefix_group[w2]['position']]['text'].lower()
          if w1 == w2 or answer != (prefix_group[w1]['lex'] == prefix_group[w2]['lex']) or t1 == t2:
            continue
          break
        feature = {'answer': answer, 'id1': prefix_group[w1]['position'], 'id2': prefix_group[w2]['position']}
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
          if wcnt[line['text'].lower()] < 1:
            wcnt[line['text'].lower()] += 1
            prefix_group.append(line)
        print 'Lines in prefix group: %d' % (len(prefix_group))

        for i in xrange(len(prefix_group)):
          for j in xrange(i+1, len(prefix_group)):
            wid1 = prefix_group[i]['position']
            wid2 = prefix_group[j]['position']
            p_key = (min(wid1, wid2), max(wid1, wid2))
            feature = {'answer': p_key in pos_ex, 'id1': wid1, 'id2': wid2}
            yield feature
    features = fgen()


  print 'Filling features...'
  not_found_vectors_count = 0
  zero_mutual_info_corpus_count = 0
  zero_mutual_info_wiki_count = 0
  for feature_no, f in enumerate(features):
    i1 = f['id1'];  w1 = words[i1]['text'];  wl1 = w1.lower()
    i2 = f['id2'];  w2 = words[i2]['text'];  wl2 = w2.lower()
    f['tag1'] = words[i1]['tag']
    f['tag2'] = words[i2]['tag']
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

    #f['log_freq_common_prefix'] = math.log(trie_fwd.get(common_prefix))
    #f['log_freq_suffix1'] = math.log(trie_inv.get(suffix1))
    #f['log_freq_suffix2'] = math.log(trie_inv.get(suffix2))

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
    save_features(args.nuts_features_output, features)


def build_croatian(args, prefixes_gen=None):
  lines = []

  sent_sets = []  # For mutual information
  word_to_sents = defaultdict(set)
  sent_set = set()

  print 'Processing CONLLX data...'
  with codecs.open(args.croatian_input, encoding='utf-8') as f:
    for line in f:
      if line.strip() == '':
        for w in sent_set:
          word_to_sents[w].add(len(sent_sets))
        sent_sets.append(sent_set)
        sent_set = set()
      else:
        lines.append(line.strip().split())
        sent_set.add(lines[-1][1].lower())

  articles = []  # wikipedia articles
  word_to_articles = defaultdict(set)
  word_to_id = {}
  id_to_word = {}
  word_to_freq = defaultdict(int)
  trie_fwd = FreqTrie(set())
  trie_inv = FreqTrie(set())

  print 'Processing wiki data...'
  with codecs.open(args.croatian_wikidata, encoding='utf-8') as f:
    for line in f:
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

  for w, f in word_to_freq.iteritems():
    trie_fwd.add(w, f)
    trie_inv.add(w[::-1], f)

  print 'Total wiki vocab size: %d' % (len(word_to_freq))
  total_wiki_words = float(sum(word_to_freq.values()))

  words = set([l[1] for l in lines])
  lemms = set([l[2] for l in lines])
  tags  = set([l[3] for l in lines])
  print ('Words: %d, lemms: %d, tags: ' + str(list(tags))) % (len(words), len(lemms))
  tagcount = defaultdict(int)
  tagsets  = defaultdict(str)
  for l in lines:
    tagcount[l[3]] += 1
    tagsets[l[2]] = ''.join(set(l[3] + tagsets[l[2]]))
  for t in tags:
    uw = len(set([l[1] for l in lines if l[3]==t]))
    ul = len(set([l[2] for l in lines if l[3]==t]))
    print "%s : %d, unique words: %d, unique lemms: %d" % (t, tagcount[t], uw, ul)
  for i, l in enumerate(lines):
    l.append(i)
  good_lines = [l for l in lines if l[3] in args.pos_tags]

  prefix_map = defaultdict(list)
  for line in lines:
    prefix_map[line[1][:3].lower()].append(line)
  prefix_map_keys = prefix_map.keys()
  print 'Prefix map croatian max pairs: %d' % sum([len(v)**2 for k, v in prefix_map.iteritems()])

  print 'Loading word2vec data...'
  vectors = wv_common.load_text_vectors(args.croatian_text_vectors)

  # 2,3,4-suffixes
  suffix_freqs = defaultdict(int)
  for w in good_lines:
    wl = w[1].lower()
    suffix_freqs[wl[-2:]] += 1
    suffix_freqs[wl[-3:]] += 1
    suffix_freqs[wl[-4:]] += 1

  suffix_freqs_total = float(sum(suffix_freqs.values()))

  word_freqs = defaultdict(int)
  for w in good_lines:
    word_freqs[w[1].lower()] += 1

  word_freqs_total = float(sum(word_freqs.values()))

  lemm_map = defaultdict(dict)
  for l in good_lines:
    lemm_map[l[2]][l[1]] = set()
  for i, l in enumerate(good_lines):
    lemm_map[l[2]][l[1]].add(i) # lemm -> forms -> positions

  positive_examples = gen_positive_examples(lemm_map)
  print "Total possible different positive examples: %d" % (len(positive_examples))

  print "Building data from %d word-positions" % len(good_lines)
  success = 0
  n_tests = args.gen_pos + args.gen_neg
  for i in xrange(n_tests):
    w1 = -1
    w2 = -1
    while w1 == w2:
      w1 = random.randint(0, len(good_lines)-1)
      w2 = random.randint(0, len(good_lines)-1)
    if good_lines[w1][2] == good_lines[w2][2]:
      success += 1
  print "Success apriori prob estimate: %f (%d successes of %d tests)" % (success * 1.0 / n_tests, success, n_tests)


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
        feature = {'answer': answer, 'id1': good_lines[w1][-1], 'id2': good_lines[w2][-1]}
      else:
        prefix_group = None
        while True:
          prefix = random.choice(prefix_map_keys)
          prefix_group = prefix_map[prefix]
          w1 = random.randint(0, len(prefix_group)-1)
          w2 = random.randint(0, len(prefix_group)-1)
          t1 = lines[prefix_group[w1][-1]][1].lower()
          t2 = lines[prefix_group[w2][-1]][1].lower()
          if w1 == w2 or answer != (prefix_group[w1][2] == prefix_group[w2][2]) or t1 == t2:
            continue
          break
        feature = {'answer': answer, 'id1': prefix_group[w1][-1], 'id2': prefix_group[w2][-1]}
      features.append(feature)
  else:
    pos_ex = set([(min(wid1, wid2), max(wid1, wid2)) for wid1, wid2 in positive_examples])
    def fgen():
      for prefix_no, prefix in enumerate(sorted(prefix_map_keys)):
        print 'Processing prefix: %s (%d of %d)' % (prefix, prefix_no, len(prefix_map_keys))
        prefix_group = []
        wcnt = defaultdict(int)
        for line in prefix_map[prefix]:
          if wcnt[line[1].lower()] < 2:
            wcnt[line[1].lower()] += 1
            prefix_group.append(line)

        for i in xrange(len(prefix_group)):
          for j in xrange(i+1, len(prefix_group)):
            wid1 = prefix_group[i][-1]
            wid2 = prefix_group[j][-1]
            p_key = (min(wid1, wid2), max(wid1, wid2))
            feature = {'answer': p_key in pos_ex, 'id1': wid1, 'id2': wid2}
            yield feature
    features = fgen()


  print 'Filling features...'

  not_found_vectors_count = 0
  zero_mutual_info_corpus_count = 0
  zero_mutual_info_wiki_count = 0
  for feature_no, f in enumerate(features):
    i1 = f['id1'];  w1 = lines[i1][1];  wl1 = w1.lower()
    i2 = f['id2'];  w2 = lines[i2][1];  wl2 = w2.lower()
    f['tag1'] = lines[i1][3]
    f['tag2'] = lines[i2][3]
    f['w1'] = w1
    f['w2'] = w2
    # TODO: add tags: "max common length", "left-pos-tag" and "right-pos-tag" for both of them
    f['common_prefix_len'] = common_prefix_len(wl1, wl2)
    f['common_prefix_len_rel'] = f['common_prefix_len'] * 2.0 / (len(w1) + len(w2))
    f['levenshtein'] = levenshtein.distance(wl1, wl2)
    f['jaro_winkler'] = levenshtein.jaro_winkler(wl1, wl2)
    f['jaro'] = levenshtein.jaro(wl1, wl2)

    common_prefix = wl1[:f['common_prefix_len']]
    suffix1 = wl1[f['common_prefix_len']:]
    suffix2 = wl2[f['common_prefix_len']:]
    #f['freq_common_prefix'] = sum([fr for w, fr in trie_fwd[common_prefix[:3]].iteritems() if id_to_word[w].startswith(common_prefix)]) / total_wiki_words
    #f['freq_suffix1']       = sum([fr for w, fr in trie_inv[suffix1[-3:]].iteritems() if id_to_word[w].endswith(suffix1)]) / total_wiki_words
    #f['freq_suffix2']       = sum([fr for w, fr in trie_inv[suffix2[-3:]].iteritems() if id_to_word[w].endswith(suffix2)]) / total_wiki_words

    f['freq_common_prefix'] = trie_fwd.get(common_prefix)  * 1.0 / trie_fwd.getsum()
    f['freq_suffix1'] = trie_inv.get(suffix1)  * 1.0 / trie_inv.getsum()
    f['freq_suffix2'] = trie_inv.get(suffix2)  * 1.0 / trie_inv.getsum()

    #print '%s %s  %f %f %f' % (wl1, wl2, f['freq_common_prefix'], f['freq_suffix1'], f['freq_suffix2'])
    #f['log_freq_common_prefix'] = math.log(trie_fwd.get(common_prefix))
    #f['log_freq_suffix1'] = math.log(trie_inv.get(suffix1))
    #f['log_freq_suffix2'] = math.log(trie_inv.get(suffix2))


    f['freq1'] = word_freqs[wl1] / word_freqs_total
    f['freq2'] = word_freqs[wl2] / word_freqs_total

    f['suf2freq1'] = suffix_freqs[wl1[-2:]] / suffix_freqs_total
    f['suf2freq2'] = suffix_freqs[wl2[-2:]] / suffix_freqs_total
    f['suf3freq1'] = suffix_freqs[wl1[-3:]] / suffix_freqs_total
    f['suf3freq2'] = suffix_freqs[wl2[-3:]] / suffix_freqs_total
    f['suf4freq1'] = suffix_freqs[wl1[-4:]] / suffix_freqs_total
    f['suf4freq2'] = suffix_freqs[wl2[-4:]] / suffix_freqs_total

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

    if feature_no % 10000 == 0:
      print 'Features processed: %d' % (feature_no)

    if prefixes_gen != None:
      yield f

  if prefixes_gen == None:
    print 'Not found word vectors for:    %d pairs of %d' % (not_found_vectors_count, len(features))
    print 'Zeroed mutual corpus info for: %d pairs of %d' % (zero_mutual_info_corpus_count, len(features))
    print 'Zeroed mutual wiki info for:   %d pairs of %d' % (zero_mutual_info_wiki_count, len(features))
    print 'Saving features...'
    save_features(args.croatian_features_output, features)

def main():
  start = datetime.datetime.now()
  random.seed()
  parser = argparse.ArgumentParser(description='Croatian and Nuts corpus loader')
  parser.add_argument('--croatian-input',  help='Croatian CONLLX corpus', default='../data/slavic/croatian.conllx')
  parser.add_argument('--nuts-corpus',  help='Russian nuts-corpus', default='../data/nuts/corpus.cpickle')
  parser.add_argument('--es-corpus', help='Spanish tagged corpus', default='../data/es/tagged_utf')

  parser.add_argument('--croatian-text-vectors',  help='Word2vec Croatian vectors in text format', default='../data/slavic/croatian_wiki_vectors.txt')
  parser.add_argument('--nuts-text-vectors',  help='Word2vec Russian vectors in text format', default='../data/nuts/word2vec_vectors.txt')
  parser.add_argument('--es-text-vectors', help='Word2vec Spanish vecs built by wiki (text format)', default='../data/es/spanish_wiki_vectors.txt')

  parser.add_argument('--croatian-wikidata', help='Croatian wikipedia data corpus', default='../data/temp/hr_data.txt')
  parser.add_argument('--russian-wikidata',  help='Russian wikipedia data corpus',  default='../data/temp/ru_data.txt')
  parser.add_argument('--spanish-wikidata',  help='Spanish wikipedia data corpus',  default='../data/es/es_data.txt')

  parser.add_argument('--pos-tags',  help='Pos-tags to take for analysis', default='ANVR')
  parser.add_argument('--gen-pos',  help='Positive examples to generate', type=int, default=80000)
  parser.add_argument('--gen-neg',  help='Negative examples to generate', type=int, default=80000)

  parser.add_argument('--croatian-features-output',  help='Learning data for croatian to save', default='../data/slavic/learn_croatian.txt')
  parser.add_argument('--nuts-features-output',  help='Learning data for russian to save', default='../data/nuts/learn_nuts.txt')
  parser.add_argument('--es-features-output',  help='Learning data for russian to save', default='../data/es/learn_es.txt')
  args = parser.parse_args()
  print 'Running with args:', args
  #list(build_croatian(args))
  list(build_nuts(args))
  list(spanish.build_spanish(args))

  finish = datetime.datetime.now()
  print 'Time to run:', finish-start


if __name__=="__main__":
  main()
