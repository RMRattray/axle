#include "trie.h"

#include <cassert>
#include <fstream>

#define IS_LCASE(x) ((x >= 'a') && (x <= 'z'))

TrieNode::TrieNode() {
    active = 0;
}

// Insert the word "word" into the trie, this node being at "from"
void TrieNode::insert(std::string& word, char * from) {
    if (*from) {
        uint32_t m = 1 << (*from - 'a' + 1);
        if (!(active & m)) {
            children[*from - 'a'] = std::make_shared<TrieNode>();
            active |= m;
        }
        children[*from - 'a']->insert(word, from + 1);
    } else {
        active |= 1;
        means = word;
    }
} 

// Check against the string at (of the form "l?n??ell??") with un-called letters given by "can"
void TrieNode::scan(std::vector<std::string> &ans, char * at, uint32_t can) const {
    switch (*at) {
        case '\0':
        {
            // reached the end of the check string - if this is a word, it's good to go
            if (active & 1) ans.push_back(means);
        }
        break;
        case '?':
        {
            // can be anything in "can" - check all existing children
            int i = 0;
            uint32_t m = 2;
            while (m < (1 << 27)) {
                if (can & m & active) children[i]->scan(ans, at + 1, can);
                m *= 2; ++i;
            }
        }
        break;
        default:
        {
            // is some other letter - check
            if (active & (1 << (*at - 'a' + 1))) children[*at - 'a']->scan(ans, at + 1, can);
        }
    }
}

Trie::Trie(std::string file) {
    std::string word;
    std::ifstream reader(file);
    while (std::getline(reader, word)) {
        for (char c : word) {
            assert(IS_LCASE(c));
        }
        root.insert(word, &word[0]);
    }
    reader.close();
}

void Trie::scan(std::vector<std::string>& ans, std::string at, std::string called) const {
    ans.clear();
    for (char c : at) {
        assert(IS_LCASE(c) || c == '?');
    }
    uint32_t m = -1;
    for (char c : called) {
        assert(IS_LCASE(c));
        m &= ~(1 << (c - 'a' + 1));
    }
    root.scan(ans, &at[0], m);
}

std::string Trie::strScan(std::string at, std::string called) const {
    std::vector<std::string> ans;
    scan(ans, at, called);
    if (ans.size() == 0) return "[]";
    std::string r = "[\"" + ans.front() + "\"";
    auto p = next(ans.begin());
    while (p != ans.end()) {
        r += ",\"" + *p + "\"";
        ++p;
    }
    return r + "]";
}