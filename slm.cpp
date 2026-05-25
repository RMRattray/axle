#include "slm.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

SmallLanguageModel::SmallLanguageModel() : gen(std::random_device{}()) {
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
        uint64_t read = 0;
        while (file.get(ch)) {
            if (ch == '\0') continue;
            if (ch >= 'A' && ch <= 'Z') ch |= 32;
            if ((ch >= 'a' && ch <= 'z') || ch == '\'') {
                last[ind] += ch;
                ran_last = false;
            } else {
                if (!ran_last) {
                    (this->*process)(last, ind);
                    ind = (ind + 1) & 3; ++read;
                    // if ((read & 4095) == 0) std::cout << "Read " << read << " tokens" << std::endl;
                    last[ind] = "";
                    ran_last = true;
                }
                if (ch != ' ') {
                    last[ind] = std::string(1, ch);
                    (this->*process)(last, ind);
                    ind = (ind + 1) & 3; ++read;
                    // if ((read & 4095) == 0) std::cout << "Read " << read << " tokens" << std::endl;
                    last[ind] = "";
                    ran_last = true;
                }
            }
        }
        if (!ran_last) (this->*process)(last, ind);
        file.close();
    } else {
        std::cout << "Failed to open file: " << text_file << std::endl;
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

void SmallLanguageModel::updateWts(std::string last[], int ind) {
    const double lr = 0.00001;   // learning rate

    std::string after[3] = { last[(ind + 1) & 3],
                             last[(ind + 2) & 3],
                             last[(ind + 3) & 3] };

    // --- 1. Predict next word using current weights ---
    std::string predicted = addWord(after);

    // --- 2. Identify the true next word ---
    std::string truth = last[ind];

    if (predicted == truth) return;  // no update needed

    // --- 3. Compute feature values for TRUE word ---
    uint32_t f_true[6];
    f_true[0] = monograms[truth];
    f_true[1] = bigrams[{after[2], truth}];
    f_true[2] = trigrams[{after[1], after[2], truth}];
    f_true[3] = attn_2[{after[1], truth}];
    f_true[4] = quadrigrams[{after[0], after[1], after[2], truth}];
    f_true[5] = attn_3[{after[0], truth}];

    // --- 4. Compute feature values for PREDICTED word ---
    uint32_t f_pred[6];
    f_pred[0] = monograms[predicted];
    f_pred[1] = bigrams[{after[2], predicted}];
    f_pred[2] = trigrams[{after[1], after[2], predicted}];
    f_pred[3] = attn_2[{after[1], predicted}];
    f_pred[4] = quadrigrams[{after[0], after[1], after[2], predicted}];
    f_pred[5] = attn_3[{after[0], predicted}];

    // --- 5. Perceptron update: move weights toward truth, away from prediction ---
    for (int i = 0; i < 6; i++) {
        weights[i] += lr * (f_true[i] - f_pred[i]);
    }
}

void SmallLanguageModel::train(std::string text_file) {
    iterateThroughTokens(text_file, &SmallLanguageModel::updateWts);
    for (int i = 0; i < 6; ++i) {
        std::cout << weights[i] << " ";
    }
    std::cout << "<- weights" << std::endl;
}

std::string SmallLanguageModel::addWord(const std::string after[]) {
    uint64_t t = 0;
    std::unordered_map<std::string, uint64_t> probs;
    for (auto& [word, count] : monograms) {
        probs[word] = weights[0] * count;
        probs[word] += weights[1] * bigrams[make_pair(after[2], word)];
        probs[word] += weights[2] * trigrams[make_tuple(after[1], after[2], word)];
        probs[word] += weights[3] * attn_2[make_pair(after[1], word)];
        probs[word] += weights[4] * quadrigrams[make_tuple(after[0], after[1], after[2], word)];
        probs[word] += weights[5] * attn_3[make_pair(after[0], word)];
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

std::shared_ptr<std::vector<std::string>> SmallLanguageModel::speak(const std::vector<std::string>& input) {
    auto r = std::make_shared<std::vector<std::string>>();
    if (input.size() < 3) return r;
    r->push_back(input[input.size() - 3]);
    r->push_back(input[input.size() - 2]);
    r->push_back(input[input.size() - 1]);
    while (r->size() < 32) {
        std::string a[3] = { r->at(r->size() - 3), r->at(r->size() - 2), r->at(r->size() - 1) };
        r->push_back(addWord(a));
    }
    return r;
}

uint64_t SmallLanguageModel::evaluate(const std::vector<std::string>& words) const {
    uint64_t r = 0;
    for (int i = 0; i < words.size(); ++i) {
        if (monograms.find(words[i]) != monograms.end())
            r += weights[0] * monograms.at(words[i]);
    }
    if (words.size() == 1) return r;
    for (int i = 0; i < words.size() - 1; ++i) {
        if (bigrams.find(make_pair(words[i], words[i + 1])) != bigrams.end())
            r += weights[1] * bigrams.at(make_pair(words[i], words[i + 1]));
    }
    if (words.size() == 2) return r;
    for (int i = 0; i < words.size() - 2; ++i) {
        if (trigrams.find(make_tuple(words[i], words[i + 1], words[i + 2])) != trigrams.end())
            r += weights[2] * trigrams.at(make_tuple(words[i], words[i + 1], words[i + 2]));
        if (attn_2.find(make_pair(words[i], words[i + 2])) != attn_2.end())
            r += weights[3] * attn_2.at(make_pair(words[i], words[i + 2]));
    }
    if (words.size() == 3) return r;
    for (int i = 0; i < words.size() - 3; ++i) {
        if (quadrigrams.find(make_tuple(words[i], words[i + 1], words[i + 2], words[i + 3])) != quadrigrams.end())
            r += weights[4] * quadrigrams.at(make_tuple(words[i], words[i + 1], words[i + 2], words[i + 3]));
        if (attn_3.find(make_pair(words[i], words[i + 3])) != attn_3.end())
            r += weights[5] * attn_3.at(make_pair(words[i], words[i + 3]));
    }
    return r;
}

void SmallLanguageModel::compress(std::vector<std::string>& words) const {
    sort(words);
    auto h = words.begin();
    while (h != words.end() && monograms.find(*h) != monograms.end() && (h - words.begin() < 12)) ++h;
    words.erase(h, words.end());
}

void SmallLanguageModel::sort(std::vector<std::string>& words) const {
    std::sort(words.begin(), words.end(), [this](auto& a, auto& b) { 
        uint64_t aa = (monograms.find(a) == monograms.end()) ? 0 : monograms.at(a);
        uint64_t bb = (monograms.find(b) == monograms.end()) ? 0 : monograms.at(b);
        return aa > bb;
    });
}

void SmallLanguageModel::writeData(std::string filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return;

    auto writeString = [&](const std::string& s) {
        file.write(s.c_str(), s.size());
        file.write("\0", 1);
    };

    // ---- MONOGRAMS ----
    uint32_t size = 0;
    for (auto& [token, count] : monograms) {
        if (count >= 12) ++size;
    }
    file.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
    for (auto& [token, count] : monograms) {
        if (count < 12) continue;
        writeString(token);
        file.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
    }

    // ---- BIGRAMS ----
    size = 0;
    for (auto& [token, count] : bigrams) {
        if (count >= 12) ++size;
    }
    file.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
    for (auto& [key, count] : bigrams) {
        if (count < 12) continue;
        writeString(key.first);
        writeString(key.second);
        file.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
    }

    // ---- TRIGRAMS ----
    size = 0;
    for (auto& [token, count] : trigrams) {
        if (count >= 12) ++size;
    }
    file.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
    for (auto& [key, count] : trigrams) {
        if (count < 12) continue;
        writeString(std::get<0>(key));
        writeString(std::get<1>(key));
        writeString(std::get<2>(key));
        file.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
    }

    // ---- ATTENTION 2 ----
    size = 0;
    for (auto& [token, count] : attn_2) {
        if (count >= 12) ++size;
    }
    file.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
    for (auto& [key, count] : attn_2) {
        if (count < 12) continue;
        writeString(key.first);
        writeString(key.second);
        file.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
    }

    // ---- QUADRIGRAMS ----
    size = 0;
    for (auto& [token, count] : quadrigrams) {
        if (count >= 12) ++size;
    }
    file.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
    for (auto& [key, count] : quadrigrams) {
        if (count < 12) continue;
        writeString(std::get<0>(key));
        writeString(std::get<1>(key));
        writeString(std::get<2>(key));
        writeString(std::get<3>(key));
        file.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
    }

    // ---- ATTENTION 3 ----
    size = 0;
    for (auto& [token, count] : attn_3) {
        if (count >= 12) ++size;
    }
    file.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
    for (auto& [key, count] : attn_3) {
        if (count < 12) continue;
        writeString(key.first);
        writeString(key.second);
        file.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
    }
}

void SmallLanguageModel::readData(std::string filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return;

    auto readString = [&](std::string& out) {
        out.clear();
        char c;
        while (file.get(c)) {
            if (c == '\0') break;
            out.push_back(c);
        }
    };

    uint32_t size;
    std::string s1, s2, s3, s4;
    uint32_t count;

    // ---- MONOGRAMS ----
    file.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
    for (uint32_t i = 0; i < size; i++) {
        readString(s1);
        file.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
        monograms[s1] = count;
    }

    // ---- BIGRAMS ----
    file.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
    for (uint32_t i = 0; i < size; i++) {
        readString(s1);
        readString(s2);
        file.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
        bigrams[{s1, s2}] = count;
    }

    // ---- TRIGRAMS ----
    file.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
    for (uint32_t i = 0; i < size; i++) {
        readString(s1);
        readString(s2);
        readString(s3);
        file.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
        trigrams[{s1, s2, s3}] = count;
    }

    // ---- ATTENTION 2 ----
    file.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
    for (uint32_t i = 0; i < size; i++) {
        readString(s1);
        readString(s2);
        file.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
        attn_2[{s1, s2}] = count;
    }

    // ---- QUADRIGRAMS ----
    file.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
    for (uint32_t i = 0; i < size; i++) {
        readString(s1);
        readString(s2);
        readString(s3);
        readString(s4);
        file.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
        quadrigrams[{s1, s2, s3, s4}] = count;
    }

    // ---- ATTENTION 3 ----
    file.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
    for (uint32_t i = 0; i < size; i++) {
        readString(s1);
        readString(s2);
        file.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
        attn_3[{s1, s2}] = count;
    }
}


