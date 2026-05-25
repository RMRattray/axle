#include "slm.h"

#include <iostream>

int main() {
    SmallLanguageModel s;
    s.gather("input.txt");
    s.writeData("info.bin");

    std::cout << s.evaluate( { "blessed", "am", "I" } ) << " " << s.evaluate( { "blessed", "are", "those" } ) << std::endl;
    auto x = s.speak({ "blessed", "are", "those" });
    for (auto& w : *x) std::cout << w << " ";
    std::cout << std::endl;
}