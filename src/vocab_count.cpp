//  Tool to extract unigram counts
//
//  GloVe: Global Vectors for Word Representation
//  Copyright (c) 2014 The Board of Trustees of
//  The Leland Stanford Junior University. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  For more information, bug reports, fixes, contact:
//    Jeffrey Pennington (jpennin@stanford.edu)
//    Christopher Manning (manning@cs.stanford.edu)
//    https://github.com/stanfordnlp/GloVe/
//    GlobalVectors@googlegroups.com
//    http://nlp.stanford.edu/projects/glove/

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "args.hpp"

typedef struct vocabulary {
    std::string word;
    long long count;
} VOCAB;

int verbose = 2; // 0, 1, or 2
long long min_count = 1; // min occurrences for inclusion in vocab
long long max_vocab = 0; // max_vocab = 0 for no limit

int get_counts() {
    long long vocab_size = 100000;
    std::string str;
    std::unordered_map<std::string, long long> vocab_hash;
    std::vector<VOCAB> vocab;
    
    std::cerr << "BUILDING VOCABULARY" << std::endl;
    long long token_count = 0;
    if (verbose > 1) {
        std::cerr << "Processed 0 tokens.";
    }
    while (std::cin >> str) {
        if (str == "<unk>") {
            std::cerr << "\nError, <unk> vector found in corpus.\nPlease remove <unk>s from your corpus (e.g. cat text8 | sed -e 's/<unk>/<raw_unk>/g' > text8.new)";
            std::terminate();
        }
        vocab_hash[str] += 1;
        if ((((++token_count)%100000) == 0) && (verbose > 1)) {
            std::cerr << "\033[11G" << token_count << " tokens.";
        }
    }
    if (verbose > 1) {
        std::cerr << "\033[0GProcessed " << token_count << " tokens" << std::endl;
    }

    vocab.reserve(vocab_size);
    for (auto &v: vocab_hash) {
        if (v.second >= min_count) {
            vocab.emplace_back(VOCAB{
                v.first, 
                v.second});
        }
    }

    std::cerr << "Counted " << vocab_hash.size() << " unique words" << std::endl;
    std::sort(vocab.begin(), vocab.end(), [](auto &a, auto &b){
        if (a.count == b.count) {
            return a.word < b.word;
        } else {
            return a.count > b.count;
        }
    });
    for (auto &v: vocab) {
        std::cout << v.word << " " << v.count << std::endl;
    }
    
    return 0;
}

int main(int argc, char **argv) {
    auto help_str = std::string(argv[1]);

    if (
        (argc == 2) &&
        (
            (help_str == "-h") || 
            (help_str == "-help") || 
            (help_str == "--help")
        )
    ) {
        std::cout
            << "Simple tool to extract unigram counts" << std::endl
            << "Author: Jeffrey Pennington (jpennin@stanford.edu)" << std::endl << std::endl
            << "Usage options:" << std::endl
            << "\t-verbose <int>" << std::endl
            << "\t\tSet verbosity: 0, 1, or 2 (default)" << std::endl
            << "\t-min-count <int>" << std::endl
            << "\t\tLower limit such that words which occur fewer than <int> times are discarded." << std::endl
            << "\nExample usage:" << std::endl
            << "./vocab_count -verbose 2 -min-count 10 < corpus.txt > vocab.txt" << std::endl;
        return 0;
    }

    int i;
    if ((i = find_arg("-verbose", argc, argv)) > 0) verbose = std::atoi(argv[i + 1]);
    if ((i = find_arg("-min-count", argc, argv)) > 0) min_count = std::atoll(argv[i + 1]);
    return get_counts();
}

