#include "crow.h"
//#include "crow_all.h"
#include "trie.h"

#include <iostream>

int main()
{
    crow::SimpleApp app; //define your crow application

    Trie t("all_words.txt");

    //define your endpoint at the root directory
    CROW_ROUTE(app, "/")([](){
        return "Hello world";
    });

    CROW_ROUTE(app, "/wheel/").methods(crow::HTTPMethod::POST)([t](const crow::request& req) {
        auto x = crow::json::load(req.body);

        if (!x) return crow::response{400};
        if (!x.has("fmt") || !x.has("cld")) return crow::response{400};

        std::string fmt = x["fmt"].s();
        std::string cld = x["cld"].s();

        for (char c : fmt) {
            if ((c < 'a' || c > 'z') && c != '?') {
                return crow::response{400};
            }
        }
        for (char c : cld) {
            if (c < 'a' || c > 'z') {
                return crow::response{400};
            }
        }

        std::string s = "{\"poss\":" + t.strScan(fmt, cld) + "}";
        return crow::response{s};
    });

    //set the port, set the app to run on multiple threads, and run the app
    app.port(18080).multithreaded().run();
}