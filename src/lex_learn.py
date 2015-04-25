#!/usr/local/bin/python
# -*- coding: utf-8 -*-

# Learn on Russian and apply to Croatian VS. Learn on Croatian and apply to Croatian

# Also build automorphology

import argparse
import codecs
import random
import cPickle
import datetime

import build_features

from collections import defaultdict
import spanish

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold

LEARN_FEATURES_IDS = [5, 6] + [7, 8] + [9, 10, 11, 12, 13, 14, 15, 16] + [17, 18, 19, 20, 21, 22] + [23] #+ [24, 25, 26]
learn_params = {'oob_score': True, 'n_estimators': 30}

def convert_features(features, answer_id=2, bin_features = [],
                     float_features=LEARN_FEATURES_IDS):
  converted = []
  for feat in features:
    converted.append([feat[f_name] for f_name in build_features.OUTPUT_FEATURES])

  return process_features(converted)

def load_features(filename):
  with codecs.open(filename, encoding='utf-8') as f:
    lines = [l.strip().split('\t') for l in f]
  return process_features(lines)

def process_features(feat_data, answer_id=2, bin_features = [],
                     float_features=LEARN_FEATURES_IDS):
  lines = feat_data
  Y = [(1 if l[answer_id] == 'True' else 0) for l in lines]
  binmap = {}
  for fid in bin_features:
    binmap[fid] = sorted(list(set([l[fid] for l in lines])))
  X = []
  for line in lines:
    feats = []
    for fid in float_features:
      feats.append(float(line[fid]))
    for fid in bin_features:
      for x in binmap[fid]:
        feats.append(1 if x==line[fid] else 0)
    X.append(feats)
  return np.array(X), np.array(Y), lines

def CV_estimate(X, Y, folds=10):
  skf = StratifiedKFold(Y, n_folds=folds)
  scores = []
  for train_idx, test_idx in skf:
    lr = LogisticRegression()
    lr = RandomForestClassifier(**learn_params)
    lr.fit(X[train_idx], Y[train_idx])
    scores.append(lr.score(X[test_idx], Y[test_idx]))
  return np.average(scores), np.std(scores)

def output_morphology(filename, morphology):
  total_forms = sum([len(v) for k, v in morphology.iteritems()])
  print 'Saving automorphology: %d items with total %d forms' % (len(morphology), total_forms)
  with codecs.open(filename, 'w', encoding='utf-8') as f:
    for k, words in sorted(morphology.items()):
      print >> f, '\t'.join([k] + list(words))

def lex_learn(args, langs):
  print "Loading nuts features..."
  nX, nY, nL = load_features(args.nuts_input)
  print "Loading croatian features..."
  cX, cY, cL = load_features(args.croatian_learn_data)
  print "Loading spanish features..."
  sX, sY, sL = load_features(args.spanish_learn_data)

  print "Learning..."
  cls_weights = {1: 1.0, 0: 1.0}

  model_ru = LogisticRegression()
  model_ru = RandomForestClassifier(**learn_params)
  model_ru.fit(nX, nY)
  cnt = 0
  for i, label in enumerate(model_ru.predict(cX)):
    if label != cY[i]:
      #print i, ' '.join(cL[i]), cX[i], cY[i], label
      cnt += 1
  model_hr = LogisticRegression()
  model_hr = RandomForestClassifier(**learn_params)
  model_hr.fit(cX, cY)
  for i, label in enumerate(model_hr.predict(nX)):
    if label != nY[i]:
      #print i, ' '.join(nL[i]), nX[i], nY[i], label
      pass

  model_es = LogisticRegression()
  model_es = RandomForestClassifier(**learn_params)
  model_es.fit(sX, sY)
  print "Total errors: %d" % cnt
  print "Rus -> Cro score:", model_ru.score(cX, cY)
  print "Cro -> Rus score:", model_hr.score(nX, nY)
  print 'OOB RU: %d' % model_ru.oob_score_
  print 'OOB HR: %d' % model_hr.oob_score_
  print 'OOB ES: %d' % model_es.oob_score_
  print model_es.get_params()
  print '--- importances ---'
  for imp, i in sorted(zip(model_es.feature_importances_, LEARN_FEATURES_IDS)):
    print '%f   %s  %d' % (imp, build_features.OUTPUT_FEATURES[i], i)
  print '-------------------'

  #print "Rus -> Rus CV estimate: %f (std %f)" % CV_estimate(nX, nY)
  #print "Cro -> Cro CV estimate: %f (std %f)" % CV_estimate(cX, cY)

  # Generating croatian morphology:
  if langs['HR']:
    hr_from_hr_morphology = defaultdict(set)
    hr_from_ru_morphology = defaultdict(set)
    hr_from_es_morphology = defaultdict(set)
    feat_group = []
    model_morph = zip([model_hr, model_ru, model_es],
                      [hr_from_hr_morphology, hr_from_ru_morphology,
                       hr_from_es_morphology])
    for feat_no, feat in enumerate(build_features.build_croatian(args, True)):
      feat_group.append(feat)
      if len(feat_group) >= 10000:
        print 'Generating HR morphology... feat %d' % (feat_no)
        convertedX, convertedY, convertedL = convert_features(feat_group)
        for model, morph in model_morph:
          for i, label in enumerate(model.predict(convertedX)):
            if label:
              w1 = feat_group[i]['w1'].lower()
              w2 = feat_group[i]['w2'].lower()
              #print 'Detected close forms: %s %s' % (w1, w2)
              morph[w1].add(w2)
              morph[w2].add(w1)
        feat_group = []

    output_morphology(args.hr_from_hr_automorphology, hr_from_hr_morphology)
    output_morphology(args.hr_from_ru_automorphology, hr_from_ru_morphology)
    output_morphology(args.hr_from_es_automorphology, hr_from_es_morphology)

  # Generating russian morphology
  if langs['RU']:
    ru_from_ru_morphology = defaultdict(set)
    ru_from_hr_morphology = defaultdict(set)
    ru_from_es_morphology = defaultdict(set)
    feat_group = []
    model_morph = zip([model_hr, model_ru, model_es],
                      [ru_from_hr_morphology, ru_from_ru_morphology,
                       ru_from_es_morphology])
    for feat_no, feat in enumerate(build_features.build_nuts(args, True)):
      feat_group.append(feat)
      if len(feat_group) >= 10000:
        print 'Generating RU morphology... feat %d' % (feat_no)
        convertedX, convertedY, convertedL = convert_features(feat_group)
        for model, morph in model_morph:
          for i, label in enumerate(model.predict(convertedX)):
            if label:
              w1 = feat_group[i]['w1'].lower()
              w2 = feat_group[i]['w2'].lower()
              #print 'Detected close forms: %s %s' % (w1, w2)
              morph[w1].add(w2)
              morph[w2].add(w1)
        '''for i, label in enumerate(model_ru.predict(convertedX)):
          if label == True:
            w1 = feat_group[i]['w1'].lower()
            w2 = feat_group[i]['w2'].lower()
            #print 'Detected close forms: %s %s' % (w1, w2)
            ru_from_ru_morphology[w1].add(w2)
            ru_from_ru_morphology[w2].add(w1)
        for i, label in enumerate(model_hr.predict(convertedX)):
          if label:
            w1 = feat_group[i]['w1'].lower()
            w2 = feat_group[i]['w2'].lower()
            #print 'Detected close forms: %s %s' % (w1, w2)
            ru_from_hr_morphology[w1].add(w2)
            ru_from_hr_morphology[w2].add(w1)'''
        feat_group = []

    output_morphology(args.ru_from_hr_automorphology, ru_from_hr_morphology)
    output_morphology(args.ru_from_ru_automorphology, ru_from_ru_morphology)
    output_morphology(args.ru_from_es_automorphology, ru_from_es_morphology)

  # Generating spanish morphology
  if langs['ES']:
    es_from_es_morphology = defaultdict(set)
    es_from_ru_morphology = defaultdict(set)
    es_from_hr_morphology = defaultdict(set)
    feat_group = []
    model_morph = zip([model_hr, model_ru, model_es],
                      [es_from_hr_morphology, es_from_ru_morphology,
                       es_from_es_morphology])
    for feat_no, feat in enumerate(spanish.build_spanish(args, True)):
      feat_group.append(feat)
      if len(feat_group) >= 10000:
        print 'Generating ES morphology... feat %d' % (feat_no)
        convertedX, convertedY, convertedL = convert_features(feat_group)
        for model, morph in model_morph:
          for i, label in enumerate(model.predict(convertedX)):
            if label:
              w1 = feat_group[i]['w1'].lower()
              w2 = feat_group[i]['w2'].lower()
              #print 'Detected close forms: %s %s' % (w1, w2)
              morph[w1].add(w2)
              morph[w2].add(w1)
        feat_group = []

    output_morphology(args.es_from_es_automorphology, es_from_es_morphology)
    output_morphology(args.es_from_ru_automorphology, es_from_ru_morphology)
    output_morphology(args.es_from_hr_automorphology, es_from_hr_morphology)

def main():
  start = datetime.datetime.now()
  random.seed()
  parser = argparse.ArgumentParser(description='Croatian-Russian learner')
  parser.add_argument('--nuts-input',  help='Learning data for russian',
                      default='../data/nuts/learn_nuts.txt')
  parser.add_argument('--croatian-learn-data',  help='Learning data for croatian',
                      default='../data/slavic/learn_croatian.txt')
  parser.add_argument('--spanish-learn-data',  help='Learning data for spanish',
                      default='../data/es/learn_es.txt')

  parser.add_argument('--hr-from-hr-automorphology',  help='Croatian automorphology save file',
                    default='../data/slavic/hr_from_hr_automorphology.txt')
  parser.add_argument('--hr-from-ru-automorphology',  help='Croatian automorphology save file',
                    default='../data/slavic/hr_from_ru_automorphology.txt')
  parser.add_argument('--hr-from-es-automorphology',  help='Croatian automorphology save file',
                    default='../data/slavic/hr_from_es_automorphology.txt')

  parser.add_argument('--ru-from-ru-automorphology',  help='Russian automorphology save file',
                    default='../data/nuts/ru_from_ru_automorphology.txt')
  parser.add_argument('--ru-from-hr-automorphology',  help='Russian automorphology save file',
                    default='../data/nuts/ru_from_hr_automorphology.txt')
  parser.add_argument('--ru-from-es-automorphology',  help='Russian automorphology save file',
                    default='../data/nuts/ru_from_es_automorphology.txt')

  parser.add_argument('--es-from-es-automorphology',  help='Spanish automorphology save file',
                    default='../data/es/es_from_es_automorphology.txt')
  parser.add_argument('--es-from-ru-automorphology',  help='Spanish automorphology save file',
                    default='../data/es/es_from_ru_automorphology.txt')
  parser.add_argument('--es-from-hr-automorphology',  help='Spanish automorphology save file',
                    default='../data/es/es_from_hr_automorphology.txt')

  # Args only for build_features
  parser.add_argument('--croatian-input',  help='Croatian CONLLX corpus', default='../data/slavic/croatian.conllx')
  parser.add_argument('--croatian-text-vectors',  help='Word2vec Croatian vectors in text format', default='../data/slavic/croatian_vectors.txt')
  parser.add_argument('--croatian-wikidata', help='Croatian wikipedia data corpus', default='../data/temp/hr_data.txt')
  parser.add_argument('--russian-wikidata',  help='Russian wikipedia data corpus',  default='../data/temp/ru_data.txt')
  parser.add_argument('--nuts-corpus',  help='Russian nuts-corpus', default='../data/nuts/corpus.cpickle')
  parser.add_argument('--nuts-text-vectors',  help='Word2vec Russian vectors in text format', default='../data/nuts/word2vec_vectors.txt')
  parser.add_argument('--es-corpus', help='Spanish tagged corpus', default='../data/es/tagged_utf')
  parser.add_argument('--es-text-vectors', help='Word2vec Spanish vecs built by wiki (text format)', default='../data/es/spanish_wiki_vectors.txt')
  parser.add_argument('--spanish-wikidata',  help='Spanish wikipedia data corpus',  default='../data/es/es_data.txt')

  parser.add_argument('--pos-tags',  help='Pos-tags to take for analysis', default='ANVR')
  parser.add_argument('--gen-pos',  help='Positive examples to generate', type=int, default=80000)
  parser.add_argument('--gen-neg',  help='Negative examples to generate', type=int, default=80000)


  args = parser.parse_args()
  print 'Running with args:', args
  lex_learn(args, {
    'RU': False,
    'HR': True,
    'ES': False
  })

  finish = datetime.datetime.now()
  print 'Time to run:', finish-start


if __name__=="__main__":
  main()
