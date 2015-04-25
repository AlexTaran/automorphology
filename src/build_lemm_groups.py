#!/usr/bin/python
# -*- coding: utf-8 -*-

# Строит по корпусу отображение слово->список форм
# Нужно чтобы мерить полноту, например.
# Ну и по сути это и есть идеальная морфология к которой стремится наша морфология.
#
# Можно конечно сделать отображение лемма->список форм, но для неизвестных языков таких лемм у нас нет и для автоморфологии тоже.

# Здесь по идее будет реализация для всех корпусов, которые у меня будут.

import argparse
import codecs
import random
import cPickle

from collections import defaultdict


def build_lemm_groups_croatian(args):
  with codecs.open(args.croatian_input, encoding='utf-8') as f:
    lines = [line.strip().split() for line in f if line.strip() != '']
  lines = [l for l in lines if l[3] in 'ANVR']
  words = set([l[1] for l in lines])
  lemms = set([l[2] for l in lines])
  tags  = set([l[3] for l in lines])

  lemm_groups = defaultdict(set)
  for l in lines:
    lemm_groups[l[2].lower()].add(l[1].lower())

  print 'Saving lemms...'
  with codecs.open(args.croatian_lemm_groups_output, 'w', encoding='utf-8') as f:
    for lemm, forms in lemm_groups.iteritems():
      for form in forms:
        print>>f, form + '\t' + str(len(forms)-1) + '\t' + '\t'.join([frm for frm in forms if frm != form])


def build_lemm_groups_nuts(args):
  print "Loading nuts corpus"
  with open(args.nuts_input) as f:
    corpus = cPickle.load(f)

  print "Flattening nuts corpus"
  # addressation example: c['texts']['push']['paragraphs'][3][1][5] -> {lex, text, gr}
  words = []
  for name, text in corpus['texts'].iteritems():
    for par in text['paragraphs']:
      for sent in par:
        for i, w in enumerate(sent):
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
          if w['tag'] != '0':
            words.append(w)
  lemm_groups = defaultdict(set)
  for w in words:
    lemm_groups[w['lex'].lower()].add(w['text'].lower())

  print 'Saving lemms...'
  with codecs.open(args.nuts_lemm_groups_output, 'w', encoding='utf-8') as f:
    for lemm, forms in lemm_groups.iteritems():
      for form in forms:
        print>>f, form + '\t' + str(len(forms)-1) + '\t' + '\t'.join([frm for frm in forms if frm != form])



def main():
  parser = argparse.ArgumentParser(description='Croatian corpus loader')
  parser.add_argument('--croatian-input',  help='Croatian CONLLX corpus', default='../data/slavic/croatian.conllx')
  parser.add_argument('--nuts-input',  help='Russian nuts-corpus', default='../data/nuts/corpus.cpickle')
  parser.add_argument('--pos-tags',  help='Pos-tags to take for analysis', default='ANVR')
  parser.add_argument('--croatian-lemm-groups-output',  help='Lemm groups for croatian to save', default='../data/slavic/lemm_groups_croatian.txt')
  parser.add_argument('--nuts-lemm-groups-output',  help='Lemm groups for russian to save', default='../data/nuts/lemm_groups_nuts.txt')
  args = parser.parse_args()
  print 'Running with args:', args
  build_lemm_groups_croatian(args)


if __name__=="__main__":
  main()
