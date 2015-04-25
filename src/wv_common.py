#!/usr/bin/python

import struct
import numpy
import codecs
import sys

def load_vectors(filename, block_size=5000):
  with open(filename) as f:
    data = f.read(block_size)
    header, data = data.split('\n', 1)
    word_count, vec_size = [int(x) for x in header.split()]
    res = {}
    block_size = max(block_size, 4 * vec_size + 100)
    for i in xrange(word_count):
      if len(data) < block_size:
        data += f.read(block_size)
      w, data = data.split(' ', 1)
      w = w.strip()
      vec = struct.unpack('f' * vec_size, data[:(4 * vec_size)])
      vec = numpy.array(vec)
      vec /= numpy.sqrt(vec.dot(vec))
      data = data[(4 * vec_size):]
      res[w] = {'pos': i, 'vec': vec}
  return res

def load_text_vectors(filename):
  with codecs.open(filename, encoding='utf-8') as f:
    lines = [line.strip() for line in f]
    header, lines = lines[0], lines[1:]
    vocab_size, vec_size = [int(x) for x in header.split()]
    res = {}
    for idx, line in enumerate(lines):
      parts = line.split()
      key, vec = parts[0], numpy.array([float(elem) for elem in parts[1:]])
      vec /= numpy.sqrt(vec.dot(vec))
      res[key] = {'pos': idx, 'vec': vec}
      if len(vec) != vec_size:
        print>>sys.stderr, "Bad vec at pos %d: length is %d instead of required %d" % (idx, len(vec), vec_size)
      if idx % 10000 == 0:
        print 'Loaded %d text vectors' % idx
    if vocab_size != len(res):
      print>>sys.stderr, "Bad vocab size: found %d vectors instead of required %d" % (len(res), vocab_size)
  return res

_cmp_first_element = lambda x: x[0]

def closest(vecs, key, N=10):
  v = vecs[key]['vec']
  clos = []
  for k, val in vecs.iteritems():
    d = v.dot(val['vec'])
    if len(clos) < N:
      clos.append( (d, k) )
      clos = sorted(clos, key=_cmp_first_element, reverse=True)
    else:
      if d >= clos[-1][0]:
        clos[-1] = (d, k)
        clos = sorted(clos, key=_cmp_first_element, reverse=True)
  return clos

def closest_to_vec(vecs, vec, N=10):
  clos = []
  for k, val in vecs.iteritems():
    d = vec.dot(val['vec'])
    if len(clos) < N:
      clos.append( (d, k) )
      clos = sorted(clos, key=_cmp_first_element, reverse=True)
    else:
      if d >= clos[-1][0]:
        clos[-1] = (d, k)
        clos = sorted(clos, key=_cmp_first_element, reverse=True)
  return clos
