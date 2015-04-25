#!/usr/bin/python
# -*- coding: utf-8 -*-

# Different helpers to work with wiki dumps

import argparse
import codecs
import random
import cPickle
import re
import nuts
import datetime

from collections import defaultdict

import xml.etree.ElementTree as ET
from HTMLParser import HTMLParser

RU_ALPHABET = u'абвгдежзийклмнопрстуфхцчшщъыьэюяё'
HR_ALPHABET = u''
ES_ALPHABET = u''

# returns list of words
def extract_words(text):
  words = []
  prev_alpha = False
  currw = ''
  for i, ch in enumerate(text):
    ia = ch.isalpha()
    if ia and not prev_alpha:
      currw = ch
    elif ia and prev_alpha:
      currw = currw + ch
    elif prev_alpha and not ia:
      words.append(currw)
    prev_alpha = ia
  return words

#returns pairs (link, link_text)
def extract_links(text):
  re1 = re.compile(r'\[\[([^\|\]\[]+)\|([^\|\]\[]+)\]\]', flags=re.UNICODE)
  re2 = re.compile(r'\[\[([^\]\[\|]+)\]\]', flags=re.UNICODE)
  pairs = []
  for l, t in re1.findall(text):
    pairs.append( (l, t) )
  for l in re2.findall(text):
    pairs.append( (l, l) )
  return pairs

class SlovenianWikiParser(HTMLParser):
  def __init__(self):
    HTMLParser.__init__(self)
    self.ts = []
    self.pages = []

  def handle_starttag(self, tag, attrs):
    self.ts.append(tag)
    curpath = '/'.join(self.ts)
    if curpath == 'mediawiki/page':
      self.curpage = {'words':[], 'links':[]}

  def handle_endtag(self, tag):
    curpath = '/'.join(self.ts)
    if curpath == 'mediawiki/page':
      if 'title' in self.curpage:
        title = self.curpage['title']
        #if  not title.startswith('Wikipedija:') and \
        #    not title.startswith('Kategorija:') and \
        #    not title.startswith('Predloga:') and \
        #    not title.startswith('MediaWiki:') and \
        #    not title.startswith('Slika:'):
        if ':' not in title:
          self.pages.append(self.curpage)
      else:
        print 'Bad page title o_O', self.curpage
    self.ts.pop()

  def handle_data(self, data):
    curpath = '/'.join(self.ts)
    if curpath == 'mediawiki/page/title':
      if '\n' in data or '\t' in data:
        print 'Strange page title: ' + data
      self.curpage['title'] = data
      self.curpage['normalized_title'] = data.strip().lower()
    elif curpath == 'mediawiki/page/revision/text':
      self.curpage['words'] = extract_words(data)
      self.curpage['links'] = extract_links(data)
    #print "Encountered some data  :", data

def normalize_title(title):
  return ' '.join(title.split())

def read_in_chunks(file_obj, chunk_size=1024 * 1024):
    while True:
        data = file_obj.read(chunk_size)
        if not data:
            break
        yield data

def wiki_to_linear(wiki_filename, linear_output_filename):
  print 'parsing  wiki from file %s ...' % wiki_filename
  parser = SlovenianWikiParser()
  with codecs.open(wiki_filename, encoding='utf-8') as f:
    for i, chunk in enumerate(read_in_chunks(f)):
      parser.feed(chunk)
      print 'Feeded chunk %d' % i

  with codecs.open(linear_output_filename, 'w', encoding='utf-8') as f:
    for page in parser.pages:
      print >> f, 'TTL\t' + '\t'.join([page['title'], page['normalized_title']])
      print >> f, 'WRD\t' + '\t'.join(page['words'])
      print >> f, 'LNK\t' + '\t'.join(['[' + l.strip().replace('\n', ' ') + '|' + t.strip().replace('\n', ' ') + ']' for l, t in page['links']])
  print 'Total articles: %d' % len(parser.pages)

def linear_to_invindex(linear_filename, invindex_filename):
  invindex = defaultdict(set) # word -> docids
  curr_docid = 0
  index_size = 0
  with codecs.open(linear_filename, encoding='utf-8') as f:
    for line_no, line in enumerate(f):
      if line.startswith('TTL\t'):
        curr_docid += 1
        title_words = extract_words(line.split('\t')[-1])
        for w in title_words:
          #print 'TTL %s' % w
          invindex[w.lower()].add(curr_docid)
          index_size += 1
      elif line.startswith('WRD\t'):
        words = line.strip().split('\t')[1:]
        for w in words:
          #print 'WRD %s' % w
          invindex[w.lower()].add(curr_docid)
          index_size += 1
      elif line.startswith('LNK\t') and line.strip() != 'LNK':
        links = [elem[1:-1].split('|') for elem in line.split('\t')[1:]]
        if any(len(l) != 2 for l in links):
          print 'Defective line: %d' % line_no
          print line
          for l in links:
            if len(l) != 2:
              print 'wrng: ', l
          for l in links:
            print 'link', ' '.join(l)
          continue
        for link, link_text in links:
          for w in extract_words(link_text):
            #print 'LNK %s' % w
            invindex[w.lower()].add(curr_docid)
            index_size += 1
      if line_no % 10000 == 0:
        print 'Lines processed = %d, DocID = %d, index keys = %d, index size = %d' % (line_no, curr_docid, len(invindex), index_size)
      #if line_no > 50000:
      #  break

  print '--- STATS ---'
  print 'Redirects: %d' % (len(invindex['redirect']))
  print 'Docs: %d' % curr_docid
  print 'Index: keys = %d size = %d' % (len(invindex), index_size)

  with codecs.open(invindex_filename, 'w', encoding='utf-8') as out:
    for key in sorted(invindex.keys()):
      docids = invindex[key]
      print >>out, '\t'.join([key] + [str(docid) for docid in sorted(docids)])

def load_invindex(invindex_filename):
  invindex = defaultdict(set)
  with codecs.open(invindex_filename, encoding='utf-8') as f:
    for line_no, line in enumerate(f):
      elems = line.strip().split('\t')
      key = elems[0]
      docids = set([int(e) for e in elems[1:]])
      invindex[key] = docids
      if line_no % 50000 == 0:
        print 'Loaded %d lines of invindex' % line_no
  return invindex

def make_query(text):
  text = ''.join([ (ch if ch not in u'()1234567890,./#«’»-' else ' ') for ch in text ])
  return ' '.join(sorted([w.lower() for w in text.split() if ':' not in w]))

# args must be sets
# returns pair: precision, recall
def precision_recall(ethalon_set, result_set):
  inters_size = float(len(ethalon_set & result_set))
  precision = 1.0 if len(result_set)  == 0 else inters_size / len(result_set)
  recall    = 1.0 if len(ethalon_set) == 0 else inters_size / len(ethalon_set)
  return (precision, recall)

def print_agg_precision_recall_f1(prefix_text, precision_sum, recall_sum, num_queries):
  agg_precision = precision_sum / num_queries
  agg_recall    = recall_sum    / num_queries
  if agg_precision + agg_recall >= 1e-4:
    f1 = 2.0 * agg_precision * agg_recall / (agg_precision + agg_recall)
    print prefix_text, 'Aggregated: Precision: %f, Recall: %f, F1: %f' % (agg_precision, agg_recall, f1)

def filter_morphology_with_alphabet(morphology, alphabet):
  if len(alphabet) == 0:
    return
  alph = set(alphabet)
  deleted_cnt = 0
  for k in morphology.keys():
    if len(set(k) & alph) == 0:
      del morphology[k]
      deleted_cnt += 1
  print 'Deleted %d keys from morphology (result has %d keys)' % (deleted_cnt, len(morphology))


# generate 'search results' from linear wiki
# nuts_morphology_filename, automorphology_filename,
# stemmer_morphology_filename, pymorphy2_morphology_filename,
# self_automorphology_filename, cross_automorphology_filename
def generate_ethalon_dataset(linear_filename, invindex_filename,
                             morphologies_filenames, symm_filenames, filter_alphabet):
  print 'Testing search quality'

  DRY_RUN = False

  print 'Reading vocab...'
  pages = {}
  word_map = {} # word -> int
  pageid_to_name = {}
  last_key = ''
  with codecs.open(linear_filename, encoding='utf-8') as f:
    for line_no, line in enumerate(f):
      elems = line.strip().split('\t')
      if elems[0] == 'WRD':
        words = set([w.lower() for w in elems[1:]])
        for w in words:
          if w not in word_map:
            word_map[w] = len(word_map)
        pages[last_key]['words'] = set([word_map[w] for w in words])
      elif elems[0] == 'TTL':
        if DRY_RUN and line_no > 400000:
          break
        last_key = normalize_title(elems[2])
        pages[last_key] = {'id': len(pages), 'links_to': set(), 'queries': set()}
        pageid_to_name[pages[last_key]['id']] = last_key
        if len(pages) % 100000 == 0:
          print 'Processed: %d pages' % len(pages)


  print 'Reading links...'
  search_results = defaultdict(set)
  queries = set()
  total_links_count = 0
  broken_links_count = 0
  with codecs.open(linear_filename, encoding='utf-8') as f:
    for line_id, line in enumerate(f):
      if line_id % 100000 == 0:
        print 'Processed %d lines, total queries: %d' % (line_id, len(queries))
      if DRY_RUN and line_id > 400000:
        break
      elems = line.strip().split('\t')
      if elems[0] == 'LNK':
        for lnk in elems[1:]:
          if '|' not in lnk:
            print 'Bad LNK entry! line id %d, line = %s' % (line_id, line)
            continue
          total_links_count += 1
          l, t = lnk.split('|')
          l = normalize_title(l[1:].lower())
          t = normalize_title(t[:-1].lower())
          qrs = [make_query(l), make_query(t)]
          for q in qrs:
            query_words = q.split()
            for w in query_words:
              if w not in word_map:
                word_map[w] = len(word_map)
            query_tuple = tuple([word_map[w] for w in query_words])
            queries.add(query_tuple)
            if l in pages:
              search_results[query_tuple].add(pages[l]['id'])
          if l in pages:
            pages[l]['links_to'].add(t)
            #pages[l]['queries']
          else:
            broken_links_count += 1
  print 'Total links: %d, broken: %d' % (total_links_count, broken_links_count)

  '''print 'Writing ethalon search results:'
  exact_cnt = 0
  with codecs.open(links_graph_filename, 'w', encoding='utf-8') as f:
    for k, v in pages.iteritems():
      print >> f, 'TTL\t' + k + '\t' + str(v['id'])
      print >> f, 'LTO\t' + '\t'.join([t for t in v['links_to']])
      if k in v['links_to']:
        exact_cnt += 1
  print 'Pages with exact links to itself: %d of %d' % (exact_cnt, len(pages))'''

  print 'Loading inverted index'
  invindex = load_invindex(invindex_filename)

  print 'Loaded invindex for %d words' % len(invindex)

  print 'Loading lemm groups' # This is special structured morphology

  morph_keys = symm_filenames.keys() + morphologies_filenames.keys()
  morphologies = {}

  for morph_key, fn in symm_filenames.iteritems():
    lemm_groups = {}
    with codecs.open(fn, encoding='utf-8') as f:
      for l in f:
        line = l.strip()
        forms = line.split('\t')
        for form in forms:
          if form not in word_map:
            word_map[form] = len(word_map)
        st = set([word_map[form] for form in forms])
        for form in forms:
          lemm_groups[form] = st
    morphologies[morph_key] = lemm_groups

  for morph_key, fn in morphologies_filenames.iteritems():
    print 'Loading morphology: %s' % morph_key
    morphology = defaultdict(set)
    with codecs.open(fn, encoding='utf-8') as f:
      lines = [line.strip().split('\t') for line in f]
      for line in lines:
        forms = [elem for elem in line[1:] if elem[0] not in '0123456789']
        key = line[0]
        for form in forms:
          if form not in word_map and form[0] not in '0123456789':
            word_map[form] = len(word_map)
        morphology[key].update([word_map[form] for form in forms])
    filter_morphology_with_alphabet(morphology, filter_alphabet)
    morphologies[morph_key] = morphology

  '''
  print 'Loading lemm morphology built by corpus'
  nuts_morphology = load_nuts_morphology(nuts_morphology_filename)
  for key in nuts_morphology.keys():
    for form in nuts_morphology[key]:
      if form not in word_map:
        word_map[form] = len(word_map)
    nuts_morphology[key] = set([word_map[form] for form in nuts_morphology[key]])
  filter_morphology_with_alphabet(nuts_morphology, filter_alphabet)

  print 'Loading automorphology'
  automorphology = defaultdict(set)
  with codecs.open(automorphology_filename, encoding='utf-8') as f:
    lines = [line.strip().split('\t') for line in f]
    for line in lines:
      for form in line[1:]:
        if form not in word_map:
          word_map[form] = len(word_map)
      automorphology[line[0]] = set([word_map[form] for form in line[1:]])
  filter_morphology_with_alphabet(automorphology, filter_alphabet)

  print 'Loading stemmer morphology'
  stemmer_morphology = defaultdict(set)
  if stemmer_morphology_filename != None:
    with codecs.open(stemmer_morphology_filename, encoding='utf-8') as f:
      for line in f:
        parts = line.strip().split('\t')
        for form in parts[2:]:
          if form not in word_map:
            word_map[form] = len(word_map)
        stemmer_morphology[parts[0]] |= set([word_map[form] for form in parts[2:]])
  filter_morphology_with_alphabet(stemmer_morphology, filter_alphabet)

  print 'Loading pymorphy2 morphology'
  pymorphy2_morphology = defaultdict(set)
  if pymorphy2_morphology_filename != None:
    with codecs.open(pymorphy2_morphology_filename, encoding='utf-8') as f:
      for line in f:
        parts = line.strip().split('\t')
        for form in parts[2:]:
          if form not in word_map:
            word_map[form] = len(word_map)
        pymorphy2_morphology[parts[0]] |= set([word_map[form] for form in parts[2:]])
  filter_morphology_with_alphabet(pymorphy2_morphology, filter_alphabet)

  print 'Loading self automorphology'
  self_automorphology = defaultdict(set)
  if self_automorphology_filename != None:
    with codecs.open(self_automorphology_filename, encoding='utf-8') as f:
      for line in f:
        parts = line.strip().split('\t')
        for form in parts[2:]:
          if form not in word_map:
            word_map[form] = len(word_map)
        self_automorphology[parts[0]] |= set([word_map[form] for form in parts[2:]])
  filter_morphology_with_alphabet(self_automorphology, filter_alphabet)

  print 'Loading cross automorphology'
  cross_automorphology = defaultdict(set)
  if cross_automorphology_filename != None:
    with codecs.open(cross_automorphology_filename, encoding='utf-8') as f:
      for line in f:
        parts = line.strip().split('\t')
        for form in parts[2:]:
          if form not in word_map:
            word_map[form] = len(word_map)
        cross_automorphology[parts[0]] |= set([word_map[form] for form in parts[2:]])
  filter_morphology_with_alphabet(cross_automorphology, filter_alphabet)'''

  print 'Generating ethalon queries expansion. Queries count: %d' % len(queries)
  precision_sum = {}
  recall_sum = {}
  non_trivial_expanded_count = {}
  for key in morph_keys + ['SI']:
    precision_sum[key] = 0.0
    recall_sum[key] = 0.0
    non_trivial_expanded_count[key] = 0.0
  '''precision_lg_sum = 0.0
  recall_lg_sum = 0.0
  precision_si_sum = 0.0
  recall_si_sum = 0.0
  precision_ne_sum = 0.0
  recall_ne_sum = 0.0
  precision_au_sum = 0.0
  recall_au_sum = 0.0
  precision_st_sum = 0.0
  recall_st_sum = 0.0
  precision_py_sum = 0.0
  recall_py_sum = 0.0
  precision_se_sum = 0.0
  recall_se_sum = 0.0
  precision_cr_sum = 0.0
  recall_cr_sum = 0.0'''
  queries = list(queries)
  random.shuffle(queries)
  inv_word_map = dict([(v, k) for k, v in word_map.iteritems()]) # int -> word
  #queries = [{'orig':q, 'words': set([q.split())} for q in queries]
  '''non_trivial_expanded_count = 0
  non_trivial_nuts_exp_count = 0
  non_trivial_auto_exp_count = 0
  non_trivial_stemmer_exp_count = 0
  non_trivial_pymorphy2_exp_count = 0
  non_trivial_self_exp_count = 0
  non_trivial_cross_exp_count = 0'''

  queries_handled = 0
  for query_number, query in enumerate(queries):
    qwords = [inv_word_map[wid] for wid in query]
    print 'Processing query %d / %d : %s' % (query_number, len(queries), ' '.join(qwords))

    def expand_query_with_morphology(qw, morph):
      return [(set([word_map[w]]) if w not in morph else (morph[w] | set([word_map[w]])) ) for w in qw]
    def is_nontrivial_expansion(ex):
      return any([len(st) != 1 for st in ex])

    expanded = {}
    for key in morph_keys:
      expanded[key] = expand_query_with_morphology(qwords, morphologies[key])
      non_trivial_expanded_count[key] += is_nontrivial_expansion(expanded[key])

    '''
    expanded_words = expand_query_with_morphology(qwords, lemm_groups)
    non_trivial_expanded_count += is_nontrivial_expansion(expanded_words)

    nuts_exp_words = expand_query_with_morphology(qwords, nuts_morphology)
    non_trivial_nuts_exp_count += is_nontrivial_expansion(nuts_exp_words)

    auto_exp_words = expand_query_with_morphology(qwords, automorphology)
    non_trivial_auto_exp_count += int(is_nontrivial_expansion(auto_exp_words))

    stemmer_exp_words   = expand_query_with_morphology(qwords, stemmer_morphology)
    non_trivial_stemmer_exp_count += int(is_nontrivial_expansion(stemmer_exp_words))

    pymorphy2_exp_words = expand_query_with_morphology(qwords, pymorphy2_morphology)
    non_trivial_pymorphy2_exp_count += int(is_nontrivial_expansion(pymorphy2_exp_words))

    self_exp_words = expand_query_with_morphology(qwords, self_automorphology)
    non_trivial_self_exp_count += int(is_nontrivial_expansion(self_exp_words))

    cross_exp_words = expand_query_with_morphology(qwords, cross_automorphology)
    non_trivial_cross_exp_count += int(is_nontrivial_expansion(cross_exp_words))'''

    print 'Ideal results: %d' % (len(search_results[query]))
    if len(search_results[query]) < 5:
      print 'Names: ' + ', '.join([pageid_to_name[pageid] for pageid in search_results[query]])

    results = {}

    results['SI'] = set([-1])
    for wid in query:
      if -1 in results['SI']:
        results['SI'] = invindex[inv_word_map[wid]]
      else:
        results['SI'] &= invindex[inv_word_map[wid]]
    print 'Found %d results on invindex (SI)' % len(results['SI'])

    for key in morph_keys:
      results[key] = set([-1])
      for wg in expanded[key]:
        wg_res = set()
        for w in wg:
          wg_res |= invindex[inv_word_map[w]]
        if -1 in results[key]:
          results[key] = wg_res
        else:
          results[key] &= wg_res
      print 'Found %d results on lemm groups in invindex' % len(results[key])

    '''
    # results with lemm groups IN INVINDEX
    results_lg = set([-1])
    for wg in expanded_words:
      wg_res = set()
      for w in wg:
        wg_res |= invindex[inv_word_map[w]]
      if -1 in results_lg:
        results_lg = wg_res
      else:
        results_lg &= wg_res
    print 'Found %d results on lemm groups in invindex' % len(results_lg)

    # Results with nuts exp

    results_ne = set([-1])
    for wg in nuts_exp_words:
      wg_res = set()
      for w in wg:
        wg_res |= invindex[inv_word_map[w]]
      if -1 in results_ne:
        results_ne = wg_res
      else:
        results_ne &= wg_res
    print 'Found %d results on annotated corpus morphology expansion in invindex' % len(results_ne)

    # results with automorphology

    results_au = set([-1])
    for wg in auto_exp_words:
      print 'results_au len = %d' % len(results_au)
      wg_res = set()
      for w in wg:
        print 'AutoExp: %s, %d, now have %d' % (inv_word_map[w], len(invindex[inv_word_map[w]]), len(wg_res))
        wg_res |= invindex[inv_word_map[w]]
      print 'Found %d pages' % len(wg_res)
      if -1 in results_au:
        results_au = wg_res
      else:
        results_au &= wg_res
    print 'Found %d results on automorphology expansion in invindex' % len(results_au)


    results_st = set([-1])
    for wg in stemmer_exp_words:
      wg_res = set()
      for w in wg:
        wg_res |= invindex[inv_word_map[w]]
      if -1 in results_st:
        results_st = wg_res
      else:
        results_st &= wg_res
    print 'Found %d results on NLTK stemmer morphology expansion in invindex' % len(results_st)

    results_py = set([-1])
    for wg in pymorphy2_exp_words:
      wg_res = set()
      for w in wg:
        wg_res |= invindex[inv_word_map[w]]
      if -1 in results_py:
        results_py = wg_res
      else:
        results_py &= wg_res
    print 'Found %d results on pymorphy2 morphology in invindex' % len(results_py)

    results_se = set([-1])
    for wg in self_exp_words:
      wg_res = set()
      for w in wg:
        wg_res |= invindex[inv_word_map[w]]
      if -1 in results_se:
        results_se = wg_res
      else:
        results_se &= wg_res
    print 'Found %d results on self automorphology in invindex' % len(results_se)

    results_cr = set([-1])
    for wg in cross_exp_words:
      wg_res = set()
      for w in wg:
        wg_res |= invindex[inv_word_map[w]]
      if -1 in results_cr:
        results_cr = wg_res
      else:
        results_cr &= wg_res
    print 'Found %d results on cross automorphology in invindex' % len(results_cr)
    '''

    # Calculating precision and recall
    #precision, recall = precision_recall(search_results[query['orig']] | results_simple, results_lg)
    ethalon_results = search_results[query] | results['NE']
    if len(ethalon_results) == 0:
      print 'No ethalon results. Continuing.'
      continue
    queries_handled += 1
    precision = {}
    recall = {}
    for key in morph_keys + ['SI']:
      precision[key], recall[key] = precision_recall(ethalon_results, results[key])
      precision_sum[key] += precision[key]
      recall_sum[key] += recall[key]
      print_agg_precision_recall_f1('%s: ' % key, precision_sum[key], recall_sum[key], queries_handled)

    for key in morph_keys + ['SI']:
      print 'Non-trivial expanded queries: %d of %d' % (non_trivial_expanded_count[key],  queries_handled)

    '''precision_lg, recall_lg = precision_recall(ethalon_results, results_lg)
    precision_si, recall_si = precision_recall(ethalon_results, results_simple)
    precision_ne, recall_ne = precision_recall(ethalon_results, results_ne)
    precision_au, recall_au = precision_recall(ethalon_results, results_au)
    precision_st, recall_st = precision_recall(ethalon_results, results_st)
    precision_py, recall_py = precision_recall(ethalon_results, results_py)
    precision_se, recall_se = precision_recall(ethalon_results, results_se)
    precision_cr, recall_cr = precision_recall(ethalon_results, results_cr)
    precision_lg_sum += precision_lg
    recall_lg_sum += recall_lg
    precision_si_sum += precision_si
    recall_si_sum += recall_si
    precision_ne_sum += precision_ne
    recall_ne_sum += recall_ne
    precision_au_sum += precision_au
    recall_au_sum += recall_au
    precision_st_sum += precision_st
    recall_st_sum += recall_st
    precision_py_sum += precision_py
    recall_py_sum += recall_py
    precision_se_sum += precision_se
    recall_se_sum += recall_se
    precision_cr_sum += precision_cr
    recall_cr_sum += recall_cr
    print_agg_precision_recall_f1('LG: ', precision_lg_sum, recall_lg_sum, queries_handled)
    print_agg_precision_recall_f1('SI: ', precision_si_sum, recall_si_sum, queries_handled)
    print_agg_precision_recall_f1('NE: ', precision_ne_sum, recall_ne_sum, queries_handled)
    print_agg_precision_recall_f1('AU: ', precision_au_sum, recall_au_sum, queries_handled)
    print_agg_precision_recall_f1('ST: ', precision_st_sum, recall_st_sum, queries_handled)
    print_agg_precision_recall_f1('PY: ', precision_py_sum, recall_py_sum, queries_handled)
    print_agg_precision_recall_f1('SE: ', precision_se_sum, recall_se_sum, queries_handled)
    print_agg_precision_recall_f1('CR: ', precision_cr_sum, recall_cr_sum, queries_handled)

    print 'Non-trivial expanded queries: %d of %d' % (non_trivial_expanded_count,  queries_handled)
    print 'Non-trivial nuts_exp queries: %d of %d' % (non_trivial_nuts_exp_count,  queries_handled)
    print 'Non-trivial auto_exp queries: %d of %d' % (non_trivial_auto_exp_count,  queries_handled)
    print 'Non-trivial self_exp queries: %d of %d' % (non_trivial_self_exp_count,  queries_handled)
    print 'Non-trivial crossexp queries: %d of %d' % (non_trivial_cross_exp_count, queries_handled)'''

def data_to_vocab(linear_filename, vocab_filename):
  vocab = defaultdict(int)
  with codecs.open(linear_filename, encoding='utf-8') as f:
    for line_number, line in enumerate(f):
      elems = line.strip().split('\t')
      wrds = []
      if elems[0] == 'WRD':
        wrds = [w.lower() for w in elems[1:]]
      elif elems[0] == 'TTL':
        wrds = elems[2].split()
        #pages[last_key] = {'id': len(pages), 'links_to': set(), 'queries': set()}
        if line_number % 10000 == 0:
          print 'Processed: %d lines' % line_number
      for word in wrds:
        vocab[word] += 1

  print 'Cleaning vocab...'
  cleaned_vocab = defaultdict(int)
  for w, cnt in vocab.iteritems():
    word = ''.join([ch for ch in w if ch in RU_ALPHABET])
    if word != '':
      cleaned_vocab[word] += cnt

  print 'Printing result...'
  with codecs.open(vocab_filename, 'w', encoding='utf-8') as f:
    data = sorted(cleaned_vocab.items())
    for w, cnt in data:
      print >> f, '\t'.join([w, str(cnt)])

  print 'Calculating letters set...'
  letters = set()
  for w in cleaned_vocab.keys():
    letters.update(w)
  print ''.join(sorted(list(letters)))

def calc_ethalon_exact(linear_filename, search_results_filename):
  pass

def nuts_to_morphology(nuts_filename, output_filename):
  print 'Loading nuts corpus'
  with open(nuts_filename) as f:
    corpus = cPickle.load(f)
  print 'Generating morphology'
  lemm_map = nuts.build_morphology(corpus)
  print 'Writing output'
  with codecs.open(output_filename, 'w', encoding='utf-8') as f:
    for k, words in sorted(lemm_map.items()):
      print >> f, '\t'.join([k[0], k[1]] + list(words))

def load_nuts_morphology(nuts_morphology_filename):
  print 'Loading nuts morphology filename'
  with codecs.open(nuts_morphology_filename, encoding='utf-8') as f:
    lines = [line.strip().split('\t') for line in f]
    morphology = defaultdict(set)
    for line in lines:
      for form in line[2:]:
        morphology[form].update(line[2:])
    return morphology

def main():
  start = datetime.datetime.now()
  random.seed()
  parser = argparse.ArgumentParser(description='Wikipedia dump loader')
  # Slovenian paths
  parser.add_argument('--slovenian-wiki', help='Slovenian wiki',
                      default='../data/wiki/slwiki-latest-pages-articles.xml')
  parser.add_argument('--slovenian-data', help='Slovenian wiki data processed',
                      default='../data/temp/sl_data.txt')
  parser.add_argument('--slovenian-links',
                      help='Slovenian links graph',
                      default='../data/temp/sl_links.txt')
  parser.add_argument('--slovenian-ethalon',
                      help='Slovenian ethalon search results',
                      default='../data/temp/sl_ethalon.txt')

  # Croatian paths
  parser.add_argument('--croatian-wiki', help='Croatian wiki',
                      default='../data/wiki/hrwiki-latest-pages-articles.xml')
  parser.add_argument('--croatian-data', help='Croatian wiki data processed',
                      default='../data/temp/hr_data.txt')
  parser.add_argument('--croatian-invindex', help='Croatian wiki invindex',
                      default='../data/temp/hr_invindex.txt')
  parser.add_argument('--croatian-links',
                      help='Croatian links graph',
                      default='../data/temp/hr_links.txt')
  parser.add_argument('--croatian-ethalon',
                      help='Croatian ethalon search results',
                      default='../data/temp/hr_ethalon.txt')
  parser.add_argument('--croatian-lemm-groups',
                      help='Croatian lemm groups generated by labelled corpus',
                      default='../data/slavic/lemm_groups_croatian.txt')
  parser.add_argument('--croatian-prefix-lemm-groups',
                      help='Croatian lemm groups generated by labelled corpus',
                      default='../data/temp/hr_lemm_groups_prefix4.txt')
  parser.add_argument('--croatian-automorphology',
                      help='Croatian automorphology',
                      default='../data/slavic/croatian_automorphology.txt')
  parser.add_argument('--hr-from-hr-automorphology',  help='Croatian automorphology save file',
                    default='../data/slavic/hr_from_hr_automorphology.txt')
  parser.add_argument('--hr-from-ru-automorphology',  help='Croatian automorphology save file',
                    default='../data/slavic/hr_from_ru_automorphology.txt')


  # Russian paths
  parser.add_argument('--russian-wiki', help='Russian wiki',
                      default='../data/wiki/ruwiki-latest-pages-articles.xml')
  parser.add_argument('--russian-data', help='Russian wiki data processed',
                      default='../data/temp/ru_data.txt')
  parser.add_argument('--russian-invindex', help='Russian wiki invindex',
                      default='../data/temp/ru_invindex.txt')
  parser.add_argument('--russian-vocab', help='Output for russian vocab generated from wikidata',
                      default='../data/temp/ru_vocab.txt')
  parser.add_argument('--russian-links',
                      help='Russian links graph',
                      default='../data/temp/ru_links.txt')
  parser.add_argument('--russian-ethalon',
                      help='Russian ethalon search results',
                      default='../data/temp/ru_ethalon.txt')
  parser.add_argument('--russian-lemm-groups',
                      help='Russian lemm groups generated by labelled corpus',
                      default='../data/nuts/lemm_groups_nuts.txt')
  parser.add_argument('--nuts',
                      help='Russian cpickle nuts',
                      default='../data/nuts/corpus.cpickle')
  parser.add_argument('--russian-nuts-morphology',
                      help='Russian nuts morphology output',
                      default='../data/temp/ru_nuts_morphology.txt')
  parser.add_argument('--russian-automorphology',
                      help='Russian automorphology',
                      default='../data/nuts/russian_automorphology.txt')
  parser.add_argument('--ru-from-ru-automorphology',
                      help='Russian automorphology',
                      default='../data/nuts/ru_from_ru_automorphology.txt')
  parser.add_argument('--ru-from-hr-automorphology',
                      help='Russian automorphology',
                      default='../data/nuts/ru_from_hr_automorphology.txt')
  parser.add_argument('--russian-stemmer-morphology',
                      help='Russian morphology built using RussianStemmer from nltk',
                      default='../data/temp/ru_stemmer_morphology.txt')
  parser.add_argument('--russian-pymorphy2-morphology',
                      help='Russian morphology built using normal_form from pymorphy2',
                      default='../data/temp/ru_pymorphy2_morphology.txt')
  parser.add_argument('--ru-mystem-morphology',  help='morphology built from mystem',
                      default='../data/temp/ru_mystem_morphology.txt')

  # Spanish paths
  parser.add_argument('--spanish-wiki', help='Spanish wiki',
                      default='../data/wiki/eswiki-latest-pages-articles.xml')
  parser.add_argument('--spanish-data', help='Spanish wiki data processed',
                      default='../data/es/es_data.txt')
  parser.add_argument('--spanish-invindex', help='Spanish wiki invindex',
                      default='../data/es/es_invindex.txt')
  parser.add_argument('--spanish-prefix-lemm-groups', help='Spanish prefix lemm groups',
                      default='../data/temp/es_prefix_lemm_groups_symm.txt')
  parser.add_argument('--spanish-corpus-morphology', help='Corpus morphology for spanish',
                      default='../data/es/corpus_morphology.txt')
  parser.add_argument('--spanish-automorphology', help='Automorphology for spanish',
                      default='../data/es/es_from_es_automorphology.txt')

  args = parser.parse_args()

  print 'Running with args:', args
  #wiki_to_linear(args.slovenian_wiki, args.slovenian_data)
  #generate_ethalon_dataset(args.slovenian_data, args.slovenian_links, args.slovenian_ethalon)

  #wiki_to_linear(args.croatian_wiki, args.croatian_data)
  #linear_to_invindex(args.croatian_data, args.croatian_invindex)
  '''generate_ethalon_dataset(args.croatian_data, args.croatian_invindex,
                           {'NE': args.croatian_lemm_groups, 'AU': args.croatian_automorphology,
                            'SE': args.hr_from_hr_automorphology, 'CR': args.hr_from_ru_automorphology},
                            {'LG': args.croatian_prefix_lemm_groups}, HR_ALPHABET)'''

  #wiki_to_linear(args.spanish_wiki, args.spanish_data)
  #linear_to_invindex(args.spanish_data, args.spanish_invindex)
  #generate_ethalon_dataset(args.spanish_data, args.spanish_invindex,
  #                         args.spanish_prefix_lemm_groups, args.spanish_corpus_morphology, args.spanish_automorphology,
  #                         None, None, ES_ALPHABET)

  #wiki_to_linear(args.russian_wiki, args.russian_data)
  #linear_to_invindex(args.russian_data, args.russian_invindex)

  generate_ethalon_dataset(args.russian_data, args.russian_invindex,
                           {'NE': args.russian_nuts_morphology, 'AU': args.russian_automorphology,
                           'ST': args.russian_stemmer_morphology, 'PY': args.russian_pymorphy2_morphology,
                           'SE': args.ru_from_ru_automorphology, 'CR': args.ru_from_hr_automorphology},
                           {'LG': args.russian_lemm_groups, 'MS': args.ru_mystem_morphology},
                           RU_ALPHABET)
  
  #data_to_vocab(args.russian_data, args.russian_vocab)
  #nuts_to_morphology(args.nuts, args.russian_nuts_morphology)
  finish = datetime.datetime.now()
  print 'Time to run:', finish-start

if __name__=="__main__":
  main()
