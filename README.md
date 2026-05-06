Crow was made to work on Linux with `sudo apt-get install libasio-dev` and by installing from the .deb file under "Releases" in the main repo

Compiled with `g++ main.cpp trie.cpp -o axle`
Currently can be tested with the curl command: `curl http://localhost:18080/wheel/ -d '{"fmt":"??ink","cld":"tink"}'`