#include "slm.h"

#include <iostream>

int main() {
    SmallLanguageModel t;
    t.readData("info.bin");
    std::cout << "Finished reading in fixed model" << std::endl;

    std::vector<std::vector<std::string>> d = {
        { "never", "gonna", "give", "you", "up" },
        { "newer", "bunny", "give", "you", "up" },
        { "never", "gonna", "give", "add", "go" },
        { "leave", "me", "alone" },
        { "leave", "him", "alone" },
        { "leave", "he", "alone" }
    };

    for (auto& p: d) {
        for (auto& w : p) std::cout << w << " ";
        std::cout << "(" << t.evaluate(p) << ")" << std::endl;
    }
}