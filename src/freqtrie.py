import datrie

class FreqTrie:
  def __init__(self, alphabet):
    self._root = {'_freq': 0}
    self._nodecount = 1
    self._datrie = datrie.Trie(alphabet)

  def add(self, word, freq):
    curr = self._root
    for ch in word:
      curr['_freq'] += freq
      if ch not in curr:
        curr[ch] = {'_freq': 0}
        self._nodecount += 1
      curr = curr[ch]
    curr['_freq'] += freq
    '''for idx in xrange(len(word) + 1):
      key = word[:idx]
      if key not in self._datrie:
        self._datrie[key] = 0
      self._datrie[key] += freq'''

  def get(self, prefix):
    #return self._datrie[prefix]
    curr = self._root
    for ch in prefix:
      if ch not in curr:
        return 0
      curr = curr[ch]
    return curr['_freq']

  def nodecount(self):
    return self._nodecount
    #return self.

  def getsum(self):
    #return self._datrie['']
    return self._root['_freq']
