#include "slm.h"

#include <iostream>
#include <filesystem>

int main() {
    SmallLanguageModelTrainer s;
    // s.gather("input.txt");
    std::string path = "./training_data"; // Your target directory
    try {
        // Check if directory exists
        if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
            // Iterate over each file in the directory
            for (const auto& entry : std::filesystem::directory_iterator(path)) {
                // Ensure we only open regular files (skip subfolders/links)
                if (std::filesystem::is_regular_file(entry.path())) {
                    s.gather(entry.path());
                    std::cout << "Finished reading: " << entry.path().filename() << std::endl;
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    s.writeData();
    std::cout << "Finished writing out fixed model" << std::endl;

    SmallLanguageModelEvaluator t;
    t.readData();
    std::cout << "Finished reading in fixed model" << std::endl;

    auto x = t.evaluateAllOptions( { 
        { "fatter", "matter", "father" },
        { "who", "why" }, 
        { "is", "as", "us" },
        { "in", "on" }
    }, 2);
    for (auto& [score, indices] : x) {
        for (auto& i : indices) {
            std::cout << i << " ";
        }
        std::cout << ": " << score << std::endl;
    }
}