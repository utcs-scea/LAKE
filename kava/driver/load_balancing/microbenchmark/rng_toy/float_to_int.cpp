
#include <stdio.h>

int main() {
    float x;
    float xx[] = {0.1, 0.2, 0.3, -0.1};

    for (float &b : xx) {

        int bb = *((int*)&b);

        printf("%d\n", bb);
    }

}