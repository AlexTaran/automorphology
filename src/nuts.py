# Helper functions to work with Nuts Corpus

import cPickle

from collections import defaultdict

def load_corpus(fn):
  with open(fn) as f:
    corpus = cPickle.load(f)
  return corpus

def iterwords(corpus):
  for text_name, text in corpus['texts'].iteritems():
    for paragraph in text['paragraphs']:
      for sentence in paragraph:
        for word in sentence:
          yield word

def build_vocab(corpus):
  vocab = defaultdict(int)
  for word in iterwords(corpus):
    vocab[word['text'].lower().replace('`', '')] += 1
  return vocab

def build_lex_vocab(corpus):
  vocab = defaultdict(int)
  for word in iterwords(corpus):
    vocab[word['lex']] += 1
  return vocab

def build_gram_vocab(corpus):
  vocab = defaultdict(int)
  for word in iterwords(corpus):
    vocab[word['gr']] += 1
  return vocab

def build_lex_to_forms(corpus):
  vocab = defaultdict(set)
  for word in iterwords(corpus):
    vocab[word['lex']].add(word['text'].lower().replace('`', ''))
  return vocab

def build_lex_to_grams(corpus):
  vocab = defaultdict(set)
  for word in iterwords(corpus):
    vocab[word['lex']].add(word['gr'])
  return vocab

def build_lex_gr_to_form(corpus):
  vocab = {}
  for word in iterwords(corpus):
    vocab[(word['lex'], word['gr'])] = word['text'].lower().replace('`', '')
  return vocab

def build_word_list_by_paragraph(corpus):
  res = []
  for text_name, text in corpus['texts'].iteritems():
    for paragraph in text['paragraphs']:
      p = []
      for sentence in paragraph:
        for word in sentence:
          p.append(word)
      res.append(p)
  return res

def build_morphology(corpus):
  lemm_map = defaultdict(set)
  words = []
  for name, text in corpus['texts'].iteritems():
    for par in text['paragraphs']:
      for sent in par:
        for word in sent:
          key = (word['lex'].lower(), word['gr'].split(',')[0].split('=')[0])
          lemm_map[key].add(word['text'].replace('`', '').lower())

  return lemm_map
