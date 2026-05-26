#include "slm.h"

#include <iostream>
#include <filesystem>

int main() {
    SmallLanguageModel s;
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