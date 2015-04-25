#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import argparse
import codecs
import random
import cPickle
import re
import pymorphy2

from collections import defaultdict
from nltk.stem.snowball import RussianStemmer

def load_vocab(filename):
  vocab = set()
  with codecs.open(filename, encoding='utf-8') as f:
    for line_id, line in enumerate(f):
      if line_id % 100000 == 0:
        print 'Processed %d lines' % line_id
      elems = line.strip().split('\t')
      if elems[0] == 'WRD':
        vocab.update(set([w.lower() for w in elems[1:]]))
  return vocab

def write_morphology(morphology, filename):
  with codecs.open(filename, 'w', encoding='utf-8') as f:
    for k, words in sorted(morphology.iteritems()):
      if len(words) > 1:
        for w in words:
          print >> f, w + '\t' + str(len(words)-1) + '\t' + '\t'.join([wf for wf in words if wf != w])

def build_stemmer_morphology(data_filename, output_filename):
  vocab = load_vocab(data_filename)

  print 'Total words in vocab: %d' % len(vocab)
  prefix_map = defaultdict(set)
  stemmer = RussianStemmer()
  for w in vocab:
    prefix_map[stemmer.stem(w)].add(w)
  print 'Total lemm groups: %d' % (len(prefix_map))
  write_morphology(prefix_map, output_filename)

def build_pymorphy2_morphology(data_filename, output_filename):
  vocab = load_vocab(data_filename)

  print 'Total words in vocab: %d' % len(vocab)
  morphology = defaultdict(set)
  analyzer = pymorphy2.MorphAnalyzer()
  for w_no, w in enumerate(vocab):
    for parsed in analyzer.parse(w):
      morphology[parsed.normal_form].add(w)
    if w_no % 1000 == 0:
      print 'Added %d words to morphology' % w_no
  print 'Total lemm groups: %d' % (len(morphology))
  write_morphology(morphology, output_filename)

def main():
  parser = argparse.ArgumentParser(description='Russian stemmer morphology')
  parser.add_argument('--data-input',  help='Data corpus in format TTL/WRD/LNK', default='../data/temp/ru_data.txt')
  parser.add_argument('--morphology-output',  help='Stemmer morphology to save', default='../data/temp/ru_stemmer_morphology.txt')
  parser.add_argument('--pymorphy2-output',  help='Pymorpyh2 morphology to save', default='../data/temp/ru_pymorphy2_morphology.txt')
  args = parser.parse_args()
  print 'Running with args:', args
  build_stemmer_morphology(args.data_input, args.morphology_output)
  build_pymorphy2_morphology(args.data_input, args.pymorphy2_output)


if __name__=="__main__":
  main()
