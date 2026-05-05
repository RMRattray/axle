#include "trie.h"
#include <vector>
#include <iostream>

int main() {
    Trie t("test_words.txt");
    std::vector<std::string> ans;
    t.scan(ans, "??ink", "ink");
    for (auto &w : ans) std::cout << w << std::endl;
    std::cout << std::endl;
    t.scan(ans, "??ink", "tink");
    for (auto &w : ans) std::cout << w << std::endl;
    return 0;
}