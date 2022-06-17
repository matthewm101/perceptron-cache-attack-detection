#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "cacheutils.h"

char arr[64 * 4096 * 64]; // 16MB, definitely enough to do prime+probe with a 2MB L2 cache
char *addr = arr + (64 * 1357);

void fr_write_zero() {
    // printf("Zero: ");
    flush(addr);
}

void frpp_write_one() {
    // printf("One: ");
    maccess(addr);
}

size_t frpp_read(size_t threshold) {
    size_t t = rdtsc();
    maccess(addr);
    t = rdtsc() - t;
    // printf("%lu\n", t);
    return t < threshold;
}

void ff_write_zero() {
    // printf("Zero: ");
    flush(addr);
}

void ff_write_one() {
    // printf("One: ");
    maccess(addr);
}

size_t ff_read(size_t threshold) {
    size_t t = rdtsc();
    flush(addr);
    t = rdtsc() - t;
    // printf("%lu\n", t);
    return t > threshold; // backwards for FF
}

void pp_write_zero() {
    // printf("Zero: ");
    for (int i = 8; i < 24; i++) {
        maccess(addr + i * 64 * 4096);
    }
}


void tx(size_t p, void (*write_one)(), void (*write_zero)()) {
    srand(1234);
    size_t t = rdtsc_nofence() + p;
    while(1) {
        if (rand() % 2) write_one(); else write_zero();
        while (rdtsc_nofence() < t);
        t += p;
    }
}

void rx(size_t p, size_t (*read)(size_t), size_t threshold) {
    size_t t = rdtsc_nofence() + p;
    volatile size_t ones = 0;
    volatile size_t zeros = 0;
    while(1) {
        if (read(threshold)) ones += 1; else zeros += 1;
        while (rdtsc_nofence() < t);
        t += p;
    }
}

void test(size_t p, void (*write_one)(), void (*write_zero)(), size_t (*read)(size_t), size_t threshold) {
    srand(1234);
    size_t t = rdtsc_nofence() + p;
    volatile size_t true_ones = 0;
    volatile size_t false_ones = 0;
    volatile size_t true_zeros = 0;
    volatile size_t false_zeros = 0;
    size_t count = 0;
    while(1) {
        if (rand() % 2) {
            write_one();
            if (read(threshold)) true_ones  += 1; else false_zeros += 1;
        } else {
            write_zero();
            if (read(threshold)) false_ones += 1; else true_zeros += 1;
        }
        while (rdtsc_nofence() < t);
        t += p;
        count += 1;
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        printf("Wrong number of arguments: ./cacheattack method mode freq threshold\n");
        printf("Method can be: 'fr', 'ff', 'pp'\n");
        printf("Mode can be: 'tx', 'rx', 'test'\n");
        printf("Freq can be any positive integer (in Hertz) or 'max'\n");
        printf("Threshold must be a positive integer (in cycles)\n");
        exit(0);
    }

    size_t threshold = atoi(argv[4]);
    if (threshold < 1) {
        printf("Bad threshold: must be positive\n");
        exit(0);
    }

    size_t freq = atoi(argv[3]);
    if (strcmp(argv[3],"max") == 0) {
        freq = 0;
    } else if (freq <= 0) {
        printf("Bad frequency: must be positive or 'max'\n");
        exit(0);
    }
    if (freq == 0) freq = 2000000000ULL;
    size_t period = 2000000000ULL / freq;

    void (*write_one)(); void (*write_zero)(); size_t (*read)(size_t);
    if (strcmp(argv[1],"fr") == 0) {
        write_one = &frpp_write_one;
        write_zero = &fr_write_zero;
        read = &frpp_read;
    } else if (strcmp(argv[1],"ff") == 0) {
        write_one = &ff_write_one;
        write_zero = &ff_write_zero;
        read = &ff_read;
    } else if (strcmp(argv[1],"pp") == 0) {
        write_one = &frpp_write_one;
        write_zero = &pp_write_zero;
        read = &frpp_read;
    } else {
        printf("Bad method: must be 'fr', 'ff', or 'pp'\n");
        exit(0);
    }

    if (strcmp(argv[2],"tx") == 0) {
        tx(period, write_one, write_zero);
    } else if (strcmp(argv[2],"rx") == 0) {
        rx(period, read, threshold);
    } else if (strcmp(argv[2],"test") == 0) {
        test(period, write_one, write_zero, read, threshold);
    } else {
        printf("Bad mode: must be 'tx', 'rx', or 'test'\n");
        exit(0);
    }
}