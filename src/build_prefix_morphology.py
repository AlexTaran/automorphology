#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import codecs
import random
import cPickle
import datetime
import re

from collections import defaultdict
import build_features_common as bfc

def load_vocab(data_filename, alphabet):
  vocab = set()
  with codecs.open(data_filename, encoding='utf-8') as f:
    for line_id, line in enumerate(f):
      if line_id % 1000000 == 0:
        print 'Processed %d lines' % line_id
      elems = line.strip().split('\t')
      if elems[0] == 'WRD':
        vocab.update(set([w.lower() for w in elems[1:]]))

  print 'Total words in vocab: %d' % len(vocab)
  for w in list(vocab):
    if len(set(w) & alphabet) == 0:
      vocab.remove(w)
  print 'Total words after filtering: %d' % len(vocab)

  return vocab

def build_prefix_morphology(data_filename, output_filename, alphabet):
  vocab = load_vocab(data_filename, alphabet)

  prefix_map = defaultdict(set)
  for w in vocab:
    prefix_len = int(len(w) * 0.6)
    '''prefix_len = 4
    if len(w) > 8:
      prefix_len = 5
    if len(w) > 10:
      prefix_len = 6'''
    prefix_map[w[:prefix_len]].add(w)
  print 'Total lemm groups: %d' % (len(prefix_map))
  with codecs.open(output_filename, 'w', encoding='utf-8') as f:
    for k, words in sorted(prefix_map.iteritems()):
      print >> f, '\t'.join(words)
      #for w in words:
      #  print >> f, w + '\t' + str(len(words)-1) + '\t' + '\t'.join([wf for wf in words if wf != w])

def build_vocab(data_filename, output_filename, alphabet):
  voc = load_vocab(data_filename, alphabet)
  with codecs.open(output_filename, 'w', encoding='utf-8') as f:
    for word in sorted(list(voc)):
      print >> f, word

def build_mystem_morphology(labelled_filename, output_filename):
  # form -> lemms
  voc_mystem = defaultdict(set)
  print 'Loading labelled data...'
  with codecs.open(labelled_filename, encoding='utf-8') as f:
    for line_no, line in enumerate(f):
      key = line.split('{')[0]
      vals = line.strip().split('{')[1][:-1].split('|')
      vals = [''.join([ch for ch in v if ch != '?']) for v in vals]
      #print ' '.join([key]+vals)
      voc_mystem[key] = set(vals)
      if line_no % 200000 == 0:
        print 'Processed %d lines' % line_no

  lemm_to_forms = defaultdict(set)
  for key, elems in voc_mystem.iteritems():
    for e in elems:
      lemm_to_forms[e].add(key)


  print 'Processing recursive merge...'
  rec_voc = {}
  for key_no, key in enumerate(voc_mystem.iterkeys()):
    forms = set([key])
    prev_len_forms = 0
    while prev_len_forms != len(forms):
      prev_len_forms = len(forms)
      lemms = set()
      for f in forms:
        lemms |= voc_mystem[f]
      forms = set()
      for l in lemms:
        forms |= lemm_to_forms[l]
    rec_voc[key] = forms
    if key_no % 100000 == 0:
      print 'Processed %d keys' % key_no


  print 'Writing output...'
  words_done = set()
  with codecs.open(output_filename, 'w', encoding='utf-8') as f:
    for lemm, forms in sorted(lemm_to_forms.iteritems()):
      if list(forms)[0] not in words_done and len(forms) != 1:
        print >> f, '\t'.join(list(forms))
      words_done |= forms

def main():
  start = datetime.datetime.now()

  parser = argparse.ArgumentParser(description='Prefix morphology builder')
  parser.add_argument('--hr-data', help='Data corpus in format TTL/WRD/LNK', default='../data/temp/hr_data.txt')
  parser.add_argument('--ru-data', help='Russian wiki data processed', default='../data/temp/ru_data.txt')
  parser.add_argument('--es-data', help='Spanish wiki data processed', default='../data/es/es_data.txt')

  parser.add_argument('--hr-morphology-output',  help='Lemm groups for croatian to save', default='../data/temp/hr_prefix_lemm_groups_symm.txt')
  parser.add_argument('--ru-morphology-output',  help='Lemm groups for russian  to save', default='../data/temp/ru_prefix_lemm_groups_symm.txt')
  parser.add_argument('--es-morphology-output',  help='Lemm groups for spanish  to save', default='../data/temp/es_prefix_lemm_groups_symm.txt')

  parser.add_argument('--ru-vocabulary-output',  help='russian vocab to save for mystem', default='../data/temp/ru_vocab.txt')
  parser.add_argument('--ru-mystem-input',       help='russian vocab labelled with mystem', default='../data/temp/ru_vocab.txt')
  parser.add_argument('--ru-mystem-output',      help='russian vocab to save for mystem', default='../data/temp/ru_vocab_mystem_labelled.txt')
  parser.add_argument('--ru-mystem-morphology',  help='morphology built from mystem data to save', default='../data/temp/ru_mystem_morphology.txt')

  args = parser.parse_args()
  print 'Running with args:', args
  #build_prefix_morphology(args.hr_data, args.hr_morphology_output, bfc.HR_ALPHABET)
  #build_prefix_morphology(args.es_data, args.es_morphology_output, bfc.ES_ALPHABET)
  #build_prefix_morphology(args.ru_data, args.ru_morphology_output, bfc.RU_ALPHABET)


  #build_vocab(args.ru_data, args.ru_vocabulary_output, bfc.RU_ALPHABET)
  build_mystem_morphology(args.ru_mystem_output, args.ru_mystem_morphology)

  finish = datetime.datetime.now()
  print 'Time to run:', finish-start


if __name__=="__main__":
  main()
