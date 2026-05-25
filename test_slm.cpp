#include "slm.h"

int main() {
    SmallLanguageModel s;
    s.gather("input.txt");
    s.writeData("info.bin");
}