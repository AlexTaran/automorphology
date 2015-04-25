#ifndef WORD2VEC_HELPERS
#define WORD2VEC_HELPERS

#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>

using std::string;
using std::unordered_map;
using std::vector;

#define MAX_STRING 100

typedef unordered_map<string, int> Vocab;

const char* END_SENT_MARKER = "</s>";

struct WordProp {
  vector<string> stems;
  vector<string> suffs;
  vector<string> all_parts;
  string word;
  int count = 0;
  bool is_short = false;
};


// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '-') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, END_SENT_MARKER);
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

Vocab FilterVocabByFreq(const Vocab& vocab, int freq) {
  Vocab result;
  for (auto& entry: vocab) {
    if (entry.second >= freq || entry.first == END_SENT_MARKER) {
      result[entry.first] = entry.second;
    }
  }
  return result;
}

Vocab FilterVocabSuffixesByFreq(const Vocab& vocab, int freq) {
  Vocab result;
  string suffix_mark = "SUF_";
  for (auto& entry: vocab) {
    bool is_suffix = (entry.first.substr(0, 4) == suffix_mark);
    if ((entry.second >= freq && entry.first.size() >= 4 && is_suffix) || !is_suffix || entry.first == END_SENT_MARKER) {
      result[entry.first] = entry.second;
    }
  }
  return result;
}

void VocabInsert(Vocab& vocab, const string& s, int count) {
  auto it = vocab.find(s);
  if (it != vocab.end()) {
    it->second += count;
  } else {
    vocab[s] = count;
  }
}

Vocab ExpandVocab(const Vocab& vocab) {
  unordered_map<string, int> result;
  for (auto& entry: vocab) {
    if (entry.first == END_SENT_MARKER) {
      result[entry.first] = entry.second;
      continue;
    }
    const string& s = entry.first;
    if (utf8::distance(s.begin(), s.end()) >= 3) {
      auto it = s.begin();
      for (utf8::advance(it, 3, s.end()); it != s.end(); utf8::next(it, s.end())) {
        int curr = std::distance(s.begin(), it);
        //VocabInsert(result, "STM_" + s.substr(0, curr), entry.second); // <- Comment here to learn only endings
        VocabInsert(result, "SUF_" + s.substr(curr), entry.second);
      }
    } else {
      VocabInsert(result, "SHR_" + s, entry.second);
    }
  }
  return result;
}

void SaveVocab(const Vocab& v, const string& filename) {
  std::ofstream ofs(filename);
  for (auto& entry: v) {
    ofs << entry.first << " " << entry.second << std::endl;
  }
}

Vocab BuildVocabFromTrainFile(const string& filename) {
  FILE *fin;
  fin = fopen(filename.c_str(), "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  Vocab result;
  result[END_SENT_MARKER] = 0;
  char word[MAX_STRING];
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    VocabInsert(result, word, 1);
  }
  fclose(fin);
  return result;
}

long long SumVocabFreqs(const Vocab& voc) {
  long long result = 0;
  for (auto& entry: voc) {
    result += entry.second;
  }
  return result;
}

unordered_map<string, WordProp> BuildWordPropsFromVocabAndExpandedVocab(const Vocab& vocab, const Vocab& expanded_vocab) {
  unordered_map<string, WordProp> result;
  printf("Reading word props..\n");
  result.clear();
  WordProp end_sent;
  end_sent.word = END_SENT_MARKER;
  end_sent.is_short = true;
  result[END_SENT_MARKER] = end_sent;
  ////
  for (auto& entry: vocab) {
    const string& s = entry.first;
    if (result.find(s) == result.end()) {
      WordProp wp;
      wp.count = entry.second;
      wp.word = s;
      if (utf8::distance(s.begin(), s.end()) >= 3) {
        wp.is_short = false;
        auto it = s.begin();
        for (utf8::advance(it, 3, s.end()); it != s.end(); utf8::next(it, s.end())) {
          int curr = std::distance(s.begin(), it);
          string stm = "STM_" + s.substr(0, curr);
          string suf = "SUF_" + s.substr(curr);
          if (expanded_vocab.find(stm) != expanded_vocab.end()) wp.stems.push_back(stm);
          if (expanded_vocab.find(suf) != expanded_vocab.end()) wp.suffs.push_back(suf);
        }
      } else {
        wp.is_short = true;
        string wrd = "SHR_" + s;
        if (expanded_vocab.find(wrd) != expanded_vocab.end()) {
          wp.stems.push_back(wrd);
          wp.suffs.push_back(wrd);
        }
      }
      result[s] = wp;
    } else {
      result[s].count += entry.second;
    }
  }
  for (auto& entry: result) {
    WordProp& prop = entry.second;
    //prop.all_parts = prop.stems; // <- comment here to learn only word endings
    if (!prop.is_short) {
      for (auto s: prop.suffs) prop.all_parts.push_back(s);
    }
  }
  return result;
}

void printVec(const vector<string> & v) {
  for (int i = 0; i < v.size(); ++i) {
    if (i != 0) printf(" ");
    printf("%s", v[i].c_str());
  }
}

#endif
