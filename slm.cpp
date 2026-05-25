#include "slm.h"

#include <fstream>
#include <iostream>

SmallLanguageModel::SmallLanguageModel() {
    total = 0;
}

// Read through file, tokenizing.  Spaces break words; apostrophes are part of words; all other non-alphabet characters are unique words
void SmallLanguageModel::iterateThroughTokens(std::string text_file, slmProcessor process) {    
    std::ifstream file(text_file);
    if (file.is_open()) {
        char ch;
        std::string last[4] = { "", "", "", "" };
        int ind = 3;
        bool ran_last = false;
        while (file.get(ch)) {
            if (ch == '\0') continue;
            if (ch >= 'A' && ch <= 'Z') ch |= 32;
            if ((ch >= 'a' && ch <= 'z') || ch == '\'') {
                last[ind] += ch;
                ran_last = false;
            } else {
                if (!ran_last) {
                    (this->*process)(last, ind);
                    ind = (ind + 1) & 3;
                    last[ind] = "";
                    ran_last = true;
                }
                if (ch != ' ') {
                    last[ind] = std::string(1, ch);
                    (this->*process)(last, ind);
                    ind = (ind + 1) & 3;
                    last[ind] = "";
                    ran_last = true;
                }
            }
        }
        if (!ran_last) (this->*process)(last, ind);
        file.close();
    }
}

void SmallLanguageModel::updateCts(std::string last[], int ind) {
    this->monograms[last[ind]]++;
    int prev = (ind + 3) & 3;
    this->bigrams[std::make_pair(last[prev], last[ind])]++;
    int anteprev = (prev + 3) & 3;
    this->trigrams[make_tuple(last[anteprev], last[prev], last[ind])]++;
    this->attn_2[std::make_pair(last[anteprev], last[ind])]++;
    int ultiprev = (anteprev + 3) & 3;
    this->quadrigrams[make_tuple(last[ultiprev], last[anteprev], last[prev], last[ind])]++;
    this->attn_3[std::make_pair(last[ultiprev], last[ind])]++;
    this->total++;
}

void SmallLanguageModel::gather(std::string text_file) {
    iterateThroughTokens(text_file, &SmallLanguageModel::updateCts);
}

void SmallLanguageModel::writeData(std::string filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        auto a = monograms.begin();
        while (a != monograms.end()) {
            // std::cout << a->first << ": " << a->second << std::endl;
            file.write(reinterpret_cast<const char*>(&(a->first[0])), a->first.size());
            file.write("\0", 1);
            file.write(reinterpret_cast<const char*>(&a->second), sizeof(uint32_t));
            a = next(a);
        }
    }
}

