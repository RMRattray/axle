#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class TrieNode {
    public:
    TrieNode();
    void insert(std::string& word, char * from);
    void scan(std::vector<std::string> &ans, char * at, uint32_t can) const;

    private:
    uint32_t active;
    std::shared_ptr<TrieNode> children[26];
    std::string means;
};

class Trie {
    public:
    Trie(std::string file);
    void scan(std::vector<std::string> &ans, std::string at, std::string called) const;
    std::string strScan(std::string at, std::string called) const;

    private:
    TrieNode root;
};