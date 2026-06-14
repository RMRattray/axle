#include "slm.h"
#include "trie.h"

#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

int main()
{
    Trie t("all_words.txt");
    SmallLanguageModelEvaluator s;
    s.readData();
    std::cout << "Established small language model" << std::endl;

    std::ifstream phrases("test_phrases.txt");
    std::string phrase;
    int ct = 0;

    while (std::getline(phrases, phrase)) {
        // For each phrase:
        std::cout << "TEST No. " << ++ct << ":  " << phrase << std::endl;

        // Chop into words, as during testing
        std::vector<std::string> words;
        {
            std::string cur;
            for (size_t i = 0; i < phrase.size(); ++i) {
                char c = phrase[i];
                c |= 32;
                bool isWordChar = (c >= 'a' && c <= 'z') || c == '_' || c == '\'';

                if (isWordChar) {
                    cur.push_back(c);
                } else {
                    if (!cur.empty()) {
                        words.push_back(cur);
                        cur.clear();
                    }
                    if (!isspace(c)) {
                        words.emplace_back(1, c);
                    }
                }
            }
            if (!cur.empty()) words.push_back(cur);
        }

        // get score for the whole string of words
        std::vector<std::vector<std::string>> poss(words.size());
        for (int i = 0; i < words.size(); ++i) poss[i].push_back(words[i]);
        double score = s.evaluateAllOptions(poss, 1).front().first;
        std::cout << "\tScore: " << score << "\n------------------------" << std::endl;

        // Search for words after reducing phrase to 'RSTLNE' and 'DMCAHO'
        poss = std::vector<std::vector<std::string>>(words.size());
        for (int i = 0; i < words.size(); ++i) {
            if (words[i][0] >= 'a' && words[i][0] <= 'z') {
                for (int j = 0; j < words[i].size(); ++j) {
                    char c = words[i][j];
                    if (!(c <= 'e' && c != 'b' || c >= 'l' && c != 'p' && c != 'q' && c <= 't' || c == 'h')) words[i][j] = '_';
                }
                t.scan(poss[i], words[i], "rstlnecmhdao");
                s.sort(poss[i]);
                s.compress(poss[i]);
            }
            else {
                poss[i].push_back(words[i]);
            }
        }

        // Evaluate phrases
        int opts = 1;
        for (auto& x: poss) opts *= x.size();
        std::cout << "RSTLNECMHDAO: " << opts << " options detected:" << std::endl;
        if (opts <= 248832 && opts > 0) {
            std::vector<std::pair<double, std::vector<int>>> vec = s.evaluateAllOptions(poss, 16);
            int i = 0;
            while (i < vec.size() && vec[i].first != score) ++i;
            std::cout << "\t" << vec.size() << " possibilities; " << i << " incorrect" << std::endl;
            std::cout << "\tBest guess: ";
            for (i = 0; i < words.size(); ++i) {
                std::string w = poss[i][vec[0].second[i]];
                if (w[0] >= 'a' && w[0] <= 'z') std::cout << " ";
                std::cout << w;
            }
            std::cout << std::endl;
        } else {
            std::cout << "\tToo many to search" << std::endl;
        }

        // Search for words after reducing phrase to 'RSTLNE'
        poss = std::vector<std::vector<std::string>>(words.size());
        for (int i = 0; i < words.size(); ++i) {
            if (words[i][0] >= 'a' && words[i][0] <= 'z' || words[i][0] == '_') {
                for (int j = 0; j < words[i].size(); ++j) {
                    char c = words[i][j];
                    if (!(c == '\'' || c == 'e' || c == 'l' || c == 'n' || c >= 'r' && c <= 't')) words[i][j] = '_';
                }
                t.scan(poss[i], words[i], "rstlne");
                s.sort(poss[i]);
                s.compress(poss[i]);
            }
            else {
                poss[i].push_back(words[i]);
            }
        }

        // Evaluate phrases
        opts = 1;
        for (auto& x: poss) opts *= x.size();
        std::cout << "RSTLNE: " << opts << " options detected:" << std::endl;
        if (opts <= 248832 && opts > 0) {
            std::vector<std::pair<double, std::vector<int>>> vec = s.evaluateAllOptions(poss, 16);
            int i = 0;
            while (i < vec.size() && vec[i].first != score) ++i;
            std::cout << "\t" << vec.size() << " possibilities; " << i << " incorrect" << std::endl;
            std::cout << "\tBest guess: ";
            for (i = 0; i < words.size(); ++i) {
                std::string w = poss[i][vec[0].second[i]];
                if (w[0] >= 'a' && w[0] <= 'z') std::cout << " ";
                std::cout << w;
            }
            std::cout << std::endl;
        } else {
            std::cout << "\tToo many to search" << std::endl;
        }

        // Search for words after reducing phrase to ''
        poss = std::vector<std::vector<std::string>>(words.size());
        for (int i = 0; i < words.size(); ++i) {
            if (words[i][0] >= 'a' && words[i][0] <= 'z' || words[i][0] == '_') {
                for (int j = 0; j < words[i].size(); ++j) {
                    words[i][j] = '_';
                }
                t.scan(poss[i], words[i], "");
                s.sort(poss[i]);
                s.compress(poss[i]);
            }
            else {
                poss[i].push_back(words[i]);
            }
        }

        // Evaluate phrases
        opts = 1;
        for (auto& x: poss) opts *= x.size();
        std::cout << "Blank: " << opts << " options detected:" << std::endl;
        if (opts <= 248832 && opts > 0) {
            std::vector<std::pair<double, std::vector<int>>> vec = s.evaluateAllOptions(poss, 16);
            int i = 0;
            while (i < vec.size() && vec[i].first != score) ++i;
            std::cout << "\t" << vec.size() << " possibilities; " << i << " incorrect" << std::endl;
            std::cout << "\tBest guess: ";
            for (i = 0; i < words.size(); ++i) {
                std::string w = poss[i][vec[0].second[i]];
                if (w[0] >= 'a' && w[0] <= 'z') std::cout << " ";
                std::cout << w;
            }
            std::cout << std::endl;
        } else {
            std::cout << "\tToo many to search" << std::endl;
        }

        std::cout << "========================" << std::endl;
    }
}
