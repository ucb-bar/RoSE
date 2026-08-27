#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <time.h>
#define ARRAY_LEN 50000

static struct timespec tm1;

static inline void start() {
    clock_gettime(CLOCK_MONOTONIC, &tm1);
}

static inline void stop() {
    struct timespec tm2;
    clock_gettime(CLOCK_MONOTONIC, &tm2);
    unsigned long long t = (tm2.tv_sec - tm1.tv_sec) * 1000 +\
                          (tm2.tv_nsec - tm1.tv_nsec) / 1000000;
    printf("%llu ms\n", t);
}

void bubble_sort (uint32_t *a, uint32_t n) {
    uint32_t i, t, s = 1;
    while (s) {
        s = 0;
        for (i = 1; i < n; i++) {
            if (a[i] < a[i - 1]) {
                t = a[i];
                a[i] = a[i - 1];
                a[i - 1] = t;
                s = 1;
            }
        }
    }
}

void generate_array(uint32_t *a) {
    for(uint32_t i=0; i<ARRAY_LEN; ++i){
        a[i] = ARRAY_LEN - i;
    }
}

void check_sorted(uint32_t *a) {
    for(uint32_t i=1; i<ARRAY_LEN; ++i){
        if(a[i] < a[i-1]){
            printf("Array is not sorted\n");
            exit(1);
        }
    }
}

void sort_array() {
    // printf("Bubble sorting array of %d elements\n", ARRAY_LEN);
    uint32_t data[ARRAY_LEN];
    generate_array(data);
    bubble_sort(data, ARRAY_LEN);
    stop();
    check_sorted(data);
}

int main(){
    start();
    sort_array();
    return 0;
}
