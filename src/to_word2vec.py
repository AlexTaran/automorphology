#!/usr/bin/python
# -*- coding: utf-8 -*-
# Helper functions to work with Croatian CONLLX Corpus

import cPickle
import argparse
import codecs

from collections import defaultdict


# Generates from croatian corpus
def generate_word2vec_learn_data_from_corpus(input_filename, output_filename):
  print 'Reading input'
  with codecs.open(input_filename, encoding='utf-8') as f:
    lines = [line.strip().split() for line in f if line.strip() != '']
  words = [l[1] for l in lines]

  print 'Writing output'
  with codecs.open(output_filename, 'w', encoding='utf-8') as f:
    print >> f, ' '.join([w.lower() for w in words])

def generate_word2vec_learn_data_from_wiki(input_filename, output_filename):
  # TTL-WRD-LNK file
  with codecs.open(input_filename, encoding='utf-8') as fin:
    with codecs.open(output_filename, 'w', encoding='utf-8') as fout:
      for line_no, line in enumerate(fin):
        elems = line.strip().split()
        if elems[0] == 'WRD':
          print >> fout, ' '.join([w.lower() for w in elems[1:]]) + ' '
        if line_no % 10000 == 0:
          print 'Processed %d lines' % line_no

def main():
  parser = argparse.ArgumentParser(description='Croatian corpus processor')
  parser.add_argument('--croatian-input',  help='Croatian CONLLX corpus', default='../data/slavic/croatian.conllx')
  parser.add_argument('--croatian-data', help='Croatian wiki data processed',
                      default='../data/temp/hr_data.txt')
  parser.add_argument('--croatian-corpus-word2vec-output',  help='Output from CONLLX corpus to learn word2vec',
                      default='../data/slavic/croatian_word2vec_input.txt')
  parser.add_argument('--croatian-data-word2vec-output',  help='Output from wikidata to learn word2vec',
                      default='../data/slavic/croatian_wiki_word2vec_input.txt')

  parser.add_argument('--spanish-data', help='Spanish wiki data processed',
                    default='../data/es/es_data.txt')
  parser.add_argument('--spanish-data-word2vec-output',  help='Output from wikidata to learn word2vec',
                      default='../data/es/spanish_wiki_word2vec_input.txt')

  args = parser.parse_args()
  print 'Running with args:', args

  #generate_word2vec_learn_data_from_corpus(args.croatian_input, args.croatian_corpus_word2vec_output)
  #generate_word2vec_learn_data_from_wiki(args.croatian_data, args.croatian_data_word2vec_output)
  generate_word2vec_learn_data_from_wiki(args.spanish_data, args.spanish_data_word2vec_output)

if __name__=="__main__":
  main()
