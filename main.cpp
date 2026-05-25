#include "crow.h"
#include "crow/middlewares/cors.h"
//#include "crow_all.h"
#include "slm.h"
#include "trie.h"

#include <iostream>
#include <unordered_map>
#include <vector>

int main()
{
    crow::App<crow::CORSHandler> app; //define your crow application

    auto& cors = app.get_middleware<crow::CORSHandler>();
    cors.global().origin("*").methods(
        crow::HTTPMethod::GET,
        crow::HTTPMethod::POST,
        crow::HTTPMethod::OPTIONS)
        .headers("Content-Type", "Authorization").allow_credentials();

    Trie t("all_words.txt");
    SmallLanguageModel s;
    s.readData("info.bin");
    std::cout << "Established small language model" << std::endl;

    //define your endpoint at the root directory
    CROW_ROUTE(app, "/")([](){
        return "Hello world";
    });

    CROW_ROUTE(app, "/wheel/").methods(crow::HTTPMethod::POST)([s, t](const crow::request& req) {
        auto x = crow::json::load(req.body);

        if (!x) return crow::response{400};
        if (!x.has("fmt") || !x.has("cld")) return crow::response{400};

        std::string fmt = x["fmt"].s();
        std::string cld = x["cld"].s();

        // Break up the "fmt" string into words.  Words must consist of lowercase letters, apostrophes, and underscores, or else be individual non-space characters
        std::vector<std::string> words;
        {
            std::string cur;
            for (size_t i = 0; i < fmt.size(); ++i) {
                char c = fmt[i];
                bool isWordChar = (c >= 'a' && c <= 'z') || c == '_' || c == '\'';

                if (isWordChar) {
                    cur.push_back(c);
                } else {
                    if (!cur.empty()) {
                        words.push_back(cur);
                        cur.clear();
                    }
                    if (!isspace(c)) {
                        words.emplace_back(1, c);
                    }
                }
            }
            if (!cur.empty()) words.push_back(cur);
        }
        
        std::vector<std::vector<std::string>> poss(words.size(), std::vector<std::string>());
        crow::json::wvalue r;
        int ii = 0;
        for (int i = 0; i < words.size(); ++i) {
            char c = words[i][0];
            if ((c >= 'a' && c <= 'z') || c == '_' || c == '\'') {
                t.scan(poss[i], words[i], cld);
                s.sort(poss[i]);
                for (int j = 0; j < poss[i].size(); ++j) {
                    r["all"][ii][j] = poss[i][j];
                }
                ++ii;
                s.compress(poss[i]);
            }
            else poss[i].push_back(words[i]); // punctuation, etc., is fixed
        }

        int opts = 1;
        for (auto& x: poss) opts *= x.size();
        std::cout << opts << " options detected" << std::endl;
        if (opts < 10000) {

            std::unordered_map<std::string, uint64_t> chances;
            std::vector<size_t> idx(poss.size(), 0);
            bool done = poss.empty();

            while (!done) {
                std::vector<std::string> chosen;
                chosen.reserve(poss.size());
                for (size_t i = 0; i < poss.size(); ++i) {
                    chosen.push_back(poss[i][idx[i]]);
                }

                std::string joined;
                for (size_t i = 0; i < chosen.size(); ++i) {
                    if (i) joined.push_back(' ');
                    joined += chosen[i];
                }

                uint64_t val = s.evaluate(chosen);
                chances[joined] = val;

                for (int i = (int)idx.size() - 1; i >= 0; --i) {
                    idx[i]++;
                    if (idx[i] < poss[i].size()) break;
                    idx[i] = 0;
                    if (i == 0) done = true;
                }
            }

            // Extract top 5
            std::vector<std::pair<std::string, uint64_t>> vec(chances.begin(), chances.end());
            std::partial_sort(vec.begin(), vec.begin() + std::min<size_t>(5, vec.size()), vec.end(),
                            [](auto& a, auto& b) { return a.second > b.second; });

            size_t count = std::min<size_t>(5, vec.size());
            for (size_t i = 0; i < count; ++i) {
                r["out"][i]["text"]  = vec[i].first;
                r["out"][i]["score"] = vec[i].second;
            }
        }

        return crow::response{r};
    });

    //set the port, set the app to run on multiple threads, and run the app
    app.port(18080).multithreaded().run();
}
