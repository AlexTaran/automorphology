#ifndef CORPUS_READER_H_
#define CORPUS_READER_H_

#include <list>
#include <vector>
#include <string>

using std::list;
using std::string;
using std::vector;

template<int BufferSize>
class CorpusReader {
 public:
  CorpusReader(int file_desc)
    : file_dsec(file_desc), buffer_pos(0);
  { }

  vector<string> next_line() {
    skip_empty_lines();
    if (
    if (lines.size() > 0) {
      string s = lines.front().pop_front();
      return s;
    }
  }

 private:
  int file_desc;
  uint8_t buffer[BufferSize];
  int32_t buffer_pos;
  list<vector<string> > lines;

  void skip_empty_lines() {
    while (lines.size() > 0 && lines.front().size() == 0) {
      lines.pop_front();
    }
  }
};

#endif
