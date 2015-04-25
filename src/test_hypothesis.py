#!/usr/bin/python

import nuts
import wv_common
import random


def S_forms(rod, an):
  for count in ['sg', 'pl']:
    for padeg in ['nom', 'gen', 'dat', 'acc', 'ins', 'loc']:
      yield 'S,%s,%s=%s,%s' %(rod, an, count, padeg)

def main():
  vectors = wv_common.load_text_vectors('../data/nuts/word2vec_vectors.txt')
  vectors = wv_common.load_text_vectors('../data/un/corpus_by_paragraph_800_xmls_simply_vectors.txt')
  lex_vectors = wv_common.load_text_vectors('../data/nuts/word2vec_vectors_lex.txt')
  print 'Vectors loaded'
  corpus = nuts.load_corpus('../data/nuts/corpus.cpickle')
  print 'Corpus loaded'

  grams = nuts.build_gram_vocab(corpus)
  lex2grams = nuts.build_lex_to_grams(corpus)
  lexgr2form = nuts.build_lex_gr_to_form(corpus)


  while True:
    N_padegs = 2
    padegs = random.sample(list(S_forms(random.choice('nmf'), random.choice(['anim', 'inan']))), 2)
    padegs_set = set(padegs)
    print "Iteration for padegs: ", padegs
    lexes = set()
    for w in nuts.iterwords(corpus):
      if (len(lex2grams[w['lex']] & padegs_set) == N_padegs and lex_vectors.has_key(w['lex'])):
        lexes.add(w['lex'])
    print "Found %d good words for this iteration: %s" % (len(lexes), ', '.join(list(lexes)[:20]))
    tries = 0
    successes = 0
    for lex in lexes:
      clos = [k for d, k in wv_common.closest(lex_vectors, lex, 10) if k in lexes][1:]
      if len(clos) ==0 :
        continue
      print "Closest to " + lex + " are " + ", ".join(clos)
      v_to = lexgr2form[(lex, padegs[1])]
      v_from = lexgr2form[(lex, padegs[0])]
      if not vectors.has_key(v_to):
        continue
      if not vectors.has_key(v_from):
        continue
      deltavec = vectors[v_to]['vec'] - vectors[v_from]['vec']
      for cl in clos:
        cl_from = lexgr2form[(cl, padegs[0])]
        if not vectors.has_key(cl_from):
          continue
        endvec = vectors[cl_from]['vec'] + deltavec
        candidates = [k for d, k in wv_common.closest_to_vec(vectors, endvec, 10)]
        success = lexgr2form[(cl, padegs[1])] in candidates
        tries += 1
        successes += int(success)
        #print "Status: %d of %d (%f)%%" % (successes, tries, successes * 100.0 / tries)
    print '-' * 100
    print 'Transition: %s to %s' % (padegs[0], padegs[1])
    if tries > 0:
      print "Status: %d of %d (%f)%%" % (successes, tries, successes * 100.0 / tries)
    print '-' * 100

  for f in S_forms():
    print f, grams[f]


if __name__ == "__main__":
  main()
