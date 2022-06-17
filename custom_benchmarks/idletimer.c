#include <stdlib.h>
#include <stdint.h>
#include "cacheutils.h"

int main(int argc, char** argv) {
    volatile size_t ms = 0;
    volatile size_t milestone = rdtsc_nofence() + 2000000;
    while(1){
        if (rdtsc_nofence() >= milestone) {
            ms += 1;
            milestone += 2000000;
        }
    }
}