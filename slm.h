#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <random>

struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
        std::size_t h1 = std::hash<std::string>{}(p.first);
        std::size_t h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 1); // combine
    }
};

struct TupleHash {
    std::size_t operator()(const std::tuple<std::string, std::string, std::string>& t) const noexcept {
        std::size_t h1 = std::hash<std::string>{}(std::get<0>(t));
        std::size_t h2 = std::hash<std::string>{}(std::get<1>(t));
        std::size_t h3 = std::hash<std::string>{}(std::get<2>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2); // combine
    }
};

struct QuadrupleHash {
    std::size_t operator()(const std::tuple<std::string, std::string, std::string, std::string>& t) const noexcept {
        std::size_t h1 = std::hash<std::string>{}(std::get<0>(t));
        std::size_t h2 = std::hash<std::string>{}(std::get<1>(t));
        std::size_t h3 = std::hash<std::string>{}(std::get<2>(t));
        std::size_t h4 = std::hash<std::string>{}(std::get<3>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3); // combine
    }
};

class SmallLanguageModel {
    public:
    typedef void (SmallLanguageModel::*slmProcessor)(std::string last[], int ind);

    SmallLanguageModel();
    void gather(std::string text_file);
    void train(std::string text_file);
    void writeData(std::string filename);
    void readData(std::string filename);
    uint64_t evaluate(const std::vector<std::string>& words);
    std::shared_ptr<std::vector<std::string>> speak(const std::vector<std::string>& input);

    private:
    void iterateThroughTokens(std::string text_file, slmProcessor process);
    void updateCts(std::string last[], int ind);
    void updateWts(std::string last[], int ind);
    std::string addWord(const std::string after[]);
    std::unordered_map<std::string, uint32_t> monograms;
    std::unordered_map<std::pair<std::string, std::string>, uint32_t, PairHash> bigrams;
    std::unordered_map<std::tuple<std::string, std::string, std::string>, uint32_t, TupleHash> trigrams;
    std::unordered_map<std::pair<std::string, std::string>, uint32_t, PairHash> attn_2;
    std::unordered_map<std::tuple<std::string, std::string, std::string, std::string>, uint32_t, QuadrupleHash> quadrigrams;
    std::unordered_map<std::pair<std::string, std::string>, uint32_t, PairHash> attn_3;
    uint64_t total = 0;
    uint64_t weights[6] = { 1, 1, 1, 1, 1, 1 };

    std::mt19937 gen; // mersenne_twister_engine seeded with rd()
};