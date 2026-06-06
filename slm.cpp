#include "slm.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <unordered_set>

#define OUTPUT_THRESHHOLD 4

enum charType {
    LETTER,
    DIGIT,
    WHITESPACE,
    PUNCTUATION
};

SmallLanguageModelTrainer::SmallLanguageModelTrainer() : gen(std::random_device{}()) {
    total = 0;
}

// Read through file, tokenizing.  Whitespace breaks tokens; apostrophes are part of words; all other non-alphabet characters are unique words
void SmallLanguageModelTrainer::iterateThroughTokens(std::string text_file, slmProcessor process) {    
    std::ifstream file(text_file);
    if (file.is_open()) {
        char ch;
        std::string last[ATTENTION + 1] = { "" };
        int ind = ATTENTION;
        charType lastChar = WHITESPACE;
        charType thisChar;
        uint64_t read = 0;
        while (file.get(ch)) {
            if (ch == '\0') continue; // Can't handle nulls in the wole

            // Determine if letter, digit, whitespace, or other
            if (ch >= 'A' && ch <= 'Z') ch |= 32;
            if ((ch >= 'a' && ch <= 'z') || ch == '\'') {
                thisChar = LETTER;
            } else if (ch >= '0' && ch <= '9') {
                thisChar = DIGIT;
            } else if (std::isspace(ch)) {
                thisChar = WHITESPACE;
            } else {
                thisChar = PUNCTUATION;
            }

            // If switching from type, or previous was a punctuation, word is new
            // if previous wasn't whitespace, process it
            if ((thisChar != lastChar || lastChar == PUNCTUATION) && lastChar != WHITESPACE) {
                (this->*process)(last, ind);
                ind++; if (ind > ATTENTION) ind = 0; ++read;
                last[ind] = "";
            }
            if (thisChar != WHITESPACE) {
                last[ind] += std::string(1, ch);
            }

            // update lastChar
            lastChar = thisChar;
        }
        if (lastChar != WHITESPACE) (this->*process)(last, ind);
        file.close();
    } else {
        std::cout << "Failed to open file: " << text_file << std::endl;
    }
}

void SmallLanguageModelTrainer::updateCts(std::string last[], int ind) {
    this->monograms[last[ind]]++;
    int prev = ind;
    int a = ATTENTION;
    while (a) {
        --a;
        --prev; if (prev < 0) prev = ATTENTION;
        polygrams[a][last[prev]][last[ind]]++;
    }
    this->total++;
}

void SmallLanguageModelTrainer::gather(std::string text_file) {
    iterateThroughTokens(text_file, &SmallLanguageModelTrainer::updateCts);
}

void SmallLanguageModelTrainer::updateWts(std::string last[], const int ind) {
    const double lr = 0.00001;   // learning rate

    // --- 1. Predict next word using current weights ---
    std::string predicted = addWord(last); // if we were doing this correctly,
    // we would adjust addWord to take ind

    // --- 2. Identify the true next word ---
    std::string truth = last[ind];

    if (predicted == truth) return;  // no update needed

    // --- 3. Compute feature values for TRUE word ---
    uint32_t f_true[ATTENTION + 1];
    f_true[0] = monograms[truth];
    int i = ind;
    for (int f = 0; f < ATTENTION; ++f) {
        --i; if (i < 0) i = ATTENTION - 1;
        f_true[f + 1] = polygrams[f][last[i]][truth];
    }

    // --- 4. Compute feature values for PREDICTED word ---
    uint32_t f_pred[ATTENTION + 1];
    f_pred[0] = monograms[predicted];
    i = ind;
    for (int f = 0; f < ATTENTION; ++f) {
        --i; if (i < 0) i = ATTENTION - 1;
        f_pred[f + 1] = polygrams[f][last[i]][predicted];
    }

    // --- 5. Perceptron update: move weights toward truth, away from prediction ---
    for (int i = 0; i <= ATTENTION; i++) {
        weights[i] += lr * (f_true[i] - f_pred[i]);
    }
}

void SmallLanguageModelTrainer::train(std::string text_file) {
    iterateThroughTokens(text_file, &SmallLanguageModelTrainer::updateWts);
    for (int i = 0; i < 6; ++i) {
        std::cout << weights[i] << " ";
    }
    std::cout << "<- weights" << std::endl;
}

// select a word to follow the first word in after[], if that word were preceded
// by the second word in after[], etc.
std::string SmallLanguageModelTrainer::addWord(const std::string after[]) {
    uint64_t t = 0;
    std::unordered_map<std::string, uint64_t> probs;
    for (auto& [word, count] : monograms) {
        probs[word] = weights[0] * count;
        for (int d = 0; d < ATTENTION; ++d) {
            probs[word] += weights[d + 1] * polygrams[d][after[d]][word];
        }
        t += probs[word];
    }

    if (t == 0) return "\n";
    
    std::uniform_int_distribution<uint64_t> distrib(0, t - 1);
    t = distrib(gen);

    for (auto& [word, prob] : probs) {
        if (prob > t) return word;
        t -= prob;
    }
    return "\n";
}

std::shared_ptr<std::vector<std::string>> SmallLanguageModelTrainer::speak(const std::vector<std::string>& input) {
    auto r = std::make_shared<std::vector<std::string>>();
    std::string a[ATTENTION] = { "" };
    for (int i = 0; i < ATTENTION && i < input.size(); ++i) {
        a[i] = input[input.size() - 1 - i];
    }
    while (r->size() < 32) {
        r->push_back(addWord(a));
        std::cout << " " << r->back();
        for (int i = ATTENTION - 1; i >= 0; --i) {
            a[i] = a[i - 1];
        }
        a[0] = r->back();
    }
    return r;
}

std::string chopWord(const std::string& word) {
    char chopped[MAX_WORD_LENGTH] = { '\0' };
    if (word.size() < MAX_WORD_LENGTH) {
        memcpy(chopped, word.c_str(), word.size());
        memset(chopped + word.size(), 0, MAX_WORD_LENGTH - word.size());
    } else {
        memcpy(chopped, word.c_str(), MAX_WORD_LENGTH);
    }
    return std::string(chopped, MAX_WORD_LENGTH);
}

void SmallLanguageModelTrainer::writeData() {
    // Open a file to write monograms
    std::ofstream word_count_file(WORD_COUNT_FILE, std::ios::binary);
    if (!word_count_file.is_open()) {
        throw std::runtime_error("Failed to open file");
        return;
    }
    std::filesystem::path dir = DATA_DIRECTORY;

    // Gather all distinct words up to a length of MAX_WORD_LENGTH
    // And keep them alphabetized so that output polygram files are alphabetized for easy seeking
    std::map<std::string, uint32_t> chopped_monograms;
    std::unordered_map<std::string, std::unordered_set<std::string>> chopped_monograms_to_real;
    char chopped[MAX_WORD_LENGTH] = { '\0' };
    for (auto& [word, count] : monograms) {
        std::string chopped = chopWord(word);
        chopped_monograms[chopped] += count;
        chopped_monograms_to_real[chopped].insert(word);
    }

    // If they appear more than once, output them to the monograms file
    for (auto& [chopped, count] : chopped_monograms) {
        if (count > OUTPUT_THRESHHOLD) {
            word_count_file.write(chopped.c_str(), MAX_WORD_LENGTH);
            word_count_file.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));

            // And produce their polygrams files, named of the form "golf2", etc
            std::string fname = std::string(chopped.c_str(), strnlen(chopped.c_str(), MAX_WORD_LENGTH)) + "X";
            for (int a = 0; a < ATTENTION; ++a) {
                fname[fname.size() - 1] = '2' + a;
                std::filesystem::path f = fname;
                std::ofstream polygram_file(dir / f, std::ios::binary);
                if (!polygram_file.is_open()) {
                    throw std::runtime_error("Failed to open file");
                    return;
                }

                for (auto& [otherchopped, options] : chopped_monograms_to_real) {
                    uint32_t total = 0;
                    for (auto& otherword : options) {
                        for (auto& thisword : chopped_monograms_to_real[chopped]) {
                            total += polygrams[ATTENTION - 1 - a][thisword][otherword];
                        }
                    }
                    if (total > OUTPUT_THRESHHOLD) {
                        polygram_file.write(otherchopped.c_str(), MAX_WORD_LENGTH);
                        polygram_file.write(reinterpret_cast<const char*>(&total), sizeof(uint32_t));
                    }
                }

                polygram_file.close();
            }
        }
    }

    word_count_file.close();
}

void SmallLanguageModelEvaluator::readData() {
    std::ifstream word_count_file(WORD_COUNT_FILE, std::ios::binary);
    if (!word_count_file.is_open()) {
        throw std::runtime_error("Failed to open file");
        return;
    }

    char s[MAX_WORD_LENGTH];
    uint32_t count;
    while (!word_count_file.eof()) {
        word_count_file.read(s, MAX_WORD_LENGTH);
        word_count_file.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
        monograms[std::string(s, MAX_WORD_LENGTH)] = count;
    }

    word_count_file.close();

    for (auto& [key, value] : monograms) {
        std::cout << key << ": " << value << std::endl;
    }
}

std::vector<std::pair<uint64_t, std::vector<int>>> SmallLanguageModelEvaluator::evaluateAllOptions(const std::vector<std::vector<std::string>>& words, int ans_ct) const {
    std::vector<std::pair<uint64_t, std::vector<int>>> r(ans_ct, make_pair(0, std::vector(words.size(), 0)));
    return r;
}

void SmallLanguageModelEvaluator::compress(std::vector<std::string>& words) const {
    sort(words);
    auto h = words.begin();
    while (h != words.end() && monograms.find(*h) != monograms.end() && (h - words.begin() < 12)) ++h;
    words.erase(h, words.end());
}

void SmallLanguageModelEvaluator::sort(std::vector<std::string>& words) const {
    std::sort(words.begin(), words.end(), [this](auto& a, auto& b) { 
        uint64_t aa = (monograms.find(a) == monograms.end()) ? 0 : monograms.at(a);
        uint64_t bb = (monograms.find(b) == monograms.end()) ? 0 : monograms.at(b);
        return aa > bb;
    });
}


