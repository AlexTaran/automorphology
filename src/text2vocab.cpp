#include <iostream>
#include <unordered_map>

using namespace std;

int main(int argc, char** argv) {
  if (argc != 1) {
    cout << "No args! Read from standard input, write to standatd output." << endl;
    return 1;
  }
  unordered_map<string, size_t> vocab;
  string input;
  size_t word_counter = 0;
  while (cin >> input) {
    ++word_counter;
    if (vocab.find(input) == vocab.end()) {
      vocab[input] = 1;
    } else {
      vocab[input] += 1;
    }
    if (word_counter % 1000000 == 0) {
      cerr << "Stats: readed " << word_counter << " words, vocab_size = " << vocab.size() << endl;
    }
  }
  cerr << "Final stats: readed " << word_counter << " words, vocab_size = " << vocab.size() << endl;
  for (auto it = vocab.begin(); it != vocab.end(); ++it) {
    cout << it->first << "\t" << it->second << endl;
  }

  return 0;
}
