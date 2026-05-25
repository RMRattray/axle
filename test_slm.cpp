#include "slm.h"

#include <iostream>

int main() {
    SmallLanguageModel s;
    s.gather("all_words.txt");
    std::cout << "Finished reading word list" << std::endl;
    s.gather("training_data/pg84.txt");
    std::cout << "Finished reading Frankenstein; or, the Modern Prometheus" << std::endl;
    s.gather("training_data/pg1184.txt");
    std::cout << "Finished reading The Count of Monte Cristo" << std::endl;
    s.gather("training_data/pg1342.txt");
    std::cout << "Finished reading Pride and Prejudice" << std::endl;
    s.gather("training_data/pg1513.txt");
    std::cout << "Finished reading Romeo and Juliet" << std::endl;
    s.gather("training_data/pg2701.txt");
    std::cout << "Finished reading Moby Dick; or, the Whale" << std::endl;
    s.writeData("info.bin");
    std::cout << "Finished writing out fixed model" << std::endl;

    SmallLanguageModel t;
    t.readData("info.bin");
    std::cout << "Finished reading in fixed model" << std::endl;

    std::cout << t.evaluate( { "blessed", "am", "I" } ) << " " << t.evaluate( { "blessed", "are", "those" } ) << std::endl;
    auto x = t.speak({ "blessed", "are", "those" });
    for (auto& w : *x) std::cout << w << " ";
    std::cout << std::endl;
}