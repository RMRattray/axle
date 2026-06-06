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

#define WORDS_TO_SPACE(x) ((x * ATTENTION + 128) / 256)

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
                last[ind] = chopWord(last[ind]);
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
        last[ind] = chopWord(last[ind]);
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

void SmallLanguageModelTrainer::getOutputThreshhold() {
    std::map<uint32_t, uint32_t> ctCts;
    for (auto& [word, count] : monograms) ctCts[count]++;
    uint32_t total = monograms.size();
    for (auto& [count, countCount] : ctCts) {
        total -= countCount;
        countCount = total;
    }

    uint64_t t = 0;
    std::cout << t << ":" << std::string(64, 'X') << " " << monograms.size() << " (~" << WORDS_TO_SPACE(monograms.size()) << " MB)" << std::endl;
    ++t;
    auto c = ctCts.begin();
    while (c != ctCts.end()) {
        uint64_t tt = t * t;
        while (c != ctCts.end() && c->first < tt) c = next(c);
        if (c != ctCts.end()) {
            std::cout << t << ":" << std::string(c->second * 64 / monograms.size(), 'X') << " " << c->second << " (~" << WORDS_TO_SPACE(c->second) << " MB)" << std::endl;
        }
        ++t;
    }
    std::cout << "\nSelect output threshhold" << std::endl;
    std::cin >> outputThreshhold;
}

void SmallLanguageModelTrainer::writeData() {
    getOutputThreshhold();
    
    // Open a file to write monograms
    std::ofstream word_count_file(WORD_COUNT_FILE, std::ios::binary);
    if (!word_count_file.is_open()) {
        throw std::runtime_error("Failed to open file");
        return;
    }
    std::filesystem::path dir = DATA_DIRECTORY;

    // If they appear more than once, output them to the monograms file
    for (auto& [word, count] : monograms) {
        if (count > outputThreshhold * outputThreshhold) {
            word_count_file.write(word.c_str(), MAX_WORD_LENGTH);
            word_count_file.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));

            // And produce their polygrams files, named of the form "golf2", etc
            std::string fname = std::string(word.c_str(), strnlen(word.c_str(), MAX_WORD_LENGTH)) + "X";
            for (int a = 0; a < ATTENTION; ++a) {

                // Convert this one's polygram entry into an *ordered* map, filtering on output threshhold
                auto& m = polygrams[ATTENTION - 1 - a][word];
                std::map<std::string, uint32_t> filtered;
                for (auto& [otherword, uscore] : monograms) {
                    if (m.find(otherword) != m.end() && m.at(otherword) > outputThreshhold) {
                        filtered[otherword] = m.at(otherword);
                    }
                }
                if (filtered.size() == 0) continue;

                fname[fname.size() - 1] = '2' + a;
                std::filesystem::path f = fname;
                std::ofstream polygram_file(dir / f, std::ios::binary);
                if (!polygram_file.is_open()) {
                    throw std::runtime_error("Failed to open file");
                    return;
                }

                for (auto& [otherword, total] : filtered) {
                    polygram_file.write(otherword.c_str(), MAX_WORD_LENGTH);
                    polygram_file.write(reinterpret_cast<const char*>(&total), sizeof(uint32_t));
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
}

std::vector<std::pair<uint64_t, std::vector<int>>> SmallLanguageModelEvaluator::evaluateAllOptions(const std::vector<std::vector<std::string>>& words, int ans_ct) const {
    std::vector<std::pair<uint64_t, std::vector<int>>> r(ans_ct, make_pair(0, std::vector(words.size(), 0)));
    std::filesystem::path dir = DATA_DIRECTORY;

    // Look up all n-gram frequencies from files:
    // For each possible first word
    std::vector<std::vector<std::vector<std::vector<uint32_t>>>> lookups(words.size());
    for (int sentenceIdx = 0; sentenceIdx < words.size(); ++sentenceIdx) {
        for (auto& rawWord : words[sentenceIdx]) {
            std::string word = chopWord(rawWord);
            std::vector<std::vector<uint32_t>> wordResults;

            // For each possible distance from it, check for polygrams
            std::string fname = std::string(word.c_str(), strnlen(word.c_str(), MAX_WORD_LENGTH)) + "X";
            for (int temporalIdx = 1; sentenceIdx + temporalIdx < words.size() && temporalIdx <= ATTENTION; ++temporalIdx) {
                std::vector<uint32_t> wordTempResults(words[sentenceIdx + temporalIdx].size(), 1);

                // Assuming such a file exists, else everything is 1
                fname[fname.size() - 1] = ('1' + temporalIdx);
                std::filesystem::path f = fname;
                std::ifstream polygram_file(dir / f, std::ios::binary);
                if (polygram_file.is_open()) {
                    polygram_file.seekg(0, polygram_file.end);
                    uint32_t otherWordCount = polygram_file.tellg() / FILE_CHUNK_LENGTH;

                    // Binary search for all other words
                    int wtrIdx = 0;
                    for (auto& otherRawword : words[sentenceIdx + temporalIdx]) {
                        std::string otherword = chopWord(otherRawword);
                        int l = 0;
                        int r = otherWordCount - 1;
                        char read[MAX_WORD_LENGTH];

                        while (r >= l) {
                            uint32_t m = (l + r) / 2;
                            polygram_file.seekg(m * FILE_CHUNK_LENGTH);
                            polygram_file.read(read, MAX_WORD_LENGTH);
                            int v = strncmp(read, otherword.c_str(), MAX_WORD_LENGTH);
                            if (v == 0) {
                                uint32_t c;
                                polygram_file.read(reinterpret_cast<char*>(&c), sizeof(uint32_t));
                                wordTempResults[wtrIdx] = c;
                                break;
                            } else if (v > 0) r = m - 1;
                            else l = m + 1;
                        }

                        ++wtrIdx;
                    }

                    polygram_file.close();
                }

                wordResults.push_back(wordTempResults);
            }

            lookups[sentenceIdx].push_back(wordResults);
        }
    }

    // Using known n-gram frequencies, evaluate all possibilities
    std::vector<int> indices(words.size(), 0);
    while (true) {
        int idxIdx;

        // Calculate score
        uint64_t score = 0;
        for (int sentenceIdx = 0; sentenceIdx < words.size(); ++sentenceIdx) {
            for (int temporalIdx = 1; sentenceIdx + temporalIdx < words.size() && temporalIdx <= ATTENTION; ++temporalIdx) {
                score += lookups[sentenceIdx][indices[sentenceIdx]][temporalIdx - 1][indices[sentenceIdx + temporalIdx]];
            }
        }

        // If score is notable, insert score into selections
        if (score > r.back().first) {
            auto t = r.begin();
            if (r.size() > 1) {
                t = prev(r.end());
                auto h = prev(t);
                while (t != r.begin() && h->first < score) {
                    *t = *h;
                    --h; --t;
                }
            }
            t->first = score;
            t->second = std::vector<int>(words.size(), 0);
            for (idxIdx = 0; idxIdx < words.size(); ++idxIdx) {
                t->second[idxIdx] = indices[idxIdx];
            }
        }

        // Update to next set of possibilities
        idxIdx = indices.size();
        while (idxIdx) {
            --idxIdx;
            indices[idxIdx]++;
            if (indices[idxIdx] == words[idxIdx].size()) indices[idxIdx] = 0;
            else break;
        }
        if (idxIdx == 0 && indices[idxIdx] == 0) break;
    }

    return r;
}

void SmallLanguageModelEvaluator::compress(std::vector<std::string>& words) const {
    sort(words);
    auto h = words.begin();
    while (h != words.end() && monograms.find(chopWord(*h)) != monograms.end() && (h - words.begin() < 12)) ++h;
    words.erase(h, words.end());
}

void SmallLanguageModelEvaluator::sort(std::vector<std::string>& words) const {
    std::sort(words.begin(), words.end(), [this](auto& aa, auto& bb) { 
        std::string a = chopWord(aa);
        std::string b = chopWord(bb);
        uint64_t aaa = (monograms.find(a) == monograms.end()) ? 0 : monograms.at(a);
        uint64_t bbb = (monograms.find(b) == monograms.end()) ? 0 : monograms.at(b);
        return aaa > bbb;
    });
}


