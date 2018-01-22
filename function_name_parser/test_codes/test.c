#include <stdio.h>

int funa(int a, int b) {
    if (a > 100) {
        return b;
    }
    return b + 30/5;
}

int main() {
    int i;
    for (i = 0; i < 10; i += 1) {
        funa(10, 5);
    }

    return 0;
}
