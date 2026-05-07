Crow was made to work on Linux with `sudo apt-get install libasio-dev` and by installing from the .deb file under "Releases" in the main repo:
`wget https://github.com/CrowCpp/Crow/releases/download/v1.3.2/Crow-1.3.2-Linux.deb`, then `sudo dpkg -i Crow-1.3.2-Linux.deb`.

Compiled with `g++ main.cpp trie.cpp -o axle`.
Currently can be tested with the curl command: `curl http://localhost:18080/wheel/ -d '{"fmt":"__ink","cld":"tink"}'`
or `curl http://localhost:18080/wheel/ -d '{"fmt":"__n_____","cld":"n"}'` for more words
or

```
fmt="___'ll"
curl http://localhost:18080/wheel/ -d "{\"fmt\":\"$fmt\",\"cld\":\"l\"}"
```

for checking apostrophe use
or `curl https://rummy.nationalrecordingregistry.net/wheel/ -d '{"fmt":"__ink","cld":"tink"}'` for the remote version.