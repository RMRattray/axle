#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <random>

#define ATTENTION 4
#define WORD_COUNT_FILE "word_counts.dat"
#define DATA_DIRECTORY "w"
#define MAX_WORD_LENGTH 12

std::string chopWord(const std::string& word);

class SmallLanguageModelTrainer {
    public:
    // Define "slmProcessor" as a pointer to a member function of an SLMTrainer
    // that returns void and takes, as arguments, an array of strings and a const int
    typedef void (SmallLanguageModelTrainer::*slmProcessor)(std::string last[], const int ind);

    SmallLanguageModelTrainer();
    void gather(std::string text_file);
    void train(std::string text_file);
    void writeData();
    std::shared_ptr<std::vector<std::string>> speak(const std::vector<std::string>& input);

    private:
    void iterateThroughTokens(std::string text_file, slmProcessor process);
    void updateCts(std::string last[], const int ind);
    void updateWts(std::string last[], const int ind);
    std::string addWord(const std::string after[]);
    std::unordered_map<std::string, uint32_t> monograms;
    std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>> polygrams[ATTENTION];

    uint64_t total = 0;
    uint64_t weights[ATTENTION + 1] = { 1, 1, 1, 1, 1 };

    std::mt19937 gen; // mersenne_twister_engine seeded with rd()
};

class SmallLanguageModelEvaluator {
    std::vector<std::pair<uint64_t, std::vector<int>>> evaluateAllOptions(const std::vector<std::vector<std::string>>& words, int ans_ct) const;
    void compress(std::vector<std::string>& words) const;
    void sort(std::vector<std::string>& words) const;
    void readData();

    private:
    std::unordered_map<std::string, uint32_t> monograms;
};