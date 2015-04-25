#!/usr/bin/python

import argparse
import codecs

from collections import defaultdict

def read_vocab(path):
  vocab = {}
  with codecs.open(path, encoding='utf-8') as f:
    for line in f:
      word, count = line.split()
      vocab[word] = int(count)
  return vocab

def write_vocab(path, vocab):
  with codecs.open(path, 'w', encoding='utf-8') as f:
    for i, word in enumerate(sorted(vocab.keys())):
      f.write('%s %d\n' % (word, vocab[word]))

def vocab_expand(args):
  vocab = read_vocab(args.vocab_input)
  print 'Input vocab size: %d' % len(vocab)

  expanded_vocab = defaultdict(int)
  for word, count in vocab.iteritems():
    for spl in xrange(3, len(word)+1):
      stem = word[:spl]
      suff = word[spl:]
      expanded_vocab['STM_' + stem] += count
      expanded_vocab['SUF_' + suff] += count
    if len(word) < 3:
      expanded_vocab['SHR_' + word] += count

  print 'Expanded vocab size: %d' % len(expanded_vocab)
  write_vocab(args.vocab_output, expanded_vocab)

def main():
  parser = argparse.ArgumentParser(description='Vocab expander by splitting the words')
  parser.add_argument('--vocab-input',  help='Vocab to expand', required=True)
  parser.add_argument('--vocab-output', help='Where to write resulting vocab', required=True)
  args = parser.parse_args()
  print 'Running with args:', args
  vocab_expand(args)

if __name__ == '__main__':
  main()
