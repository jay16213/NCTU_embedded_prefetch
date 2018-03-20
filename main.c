#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#if defined(ARM) || defined(ARM_PRE)
#include <arm_neon.h>
#include "impl_arm.c"
#endif

#define TEST_W 4096
#define TEST_H 4096


static long diff_in_us(struct timespec t1, struct timespec t2)
{
    struct timespec diff;
    if (t2.tv_nsec-t1.tv_nsec < 0) {
        diff.tv_sec  = t2.tv_sec - t1.tv_sec - 1;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec + 1000000000;
    } else {
        diff.tv_sec  = t2.tv_sec - t1.tv_sec;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec;
    }
    return (diff.tv_sec * 1000000.0 + diff.tv_nsec / 1000.0);
}



int main()
{
    struct timespec start, end;
    int *src  = (int *) malloc(sizeof(int) * TEST_W * TEST_H);
    srand(time(NULL));
    for (int y = 0; y < TEST_H; y++)
        for (int x = 0; x < TEST_W; x++)
            *(src + y * TEST_W + x) = rand();
    int *out = (int *) malloc(sizeof(int) * TEST_W * TEST_H);
#ifdef ARM
    clock_gettime(CLOCK_REALTIME, &start);
    neon_transpose(src, out, TEST_W, TEST_H);
    clock_gettime(CLOCK_REALTIME, &end);
    if(!transpose_verify(src, out, TEST_W, TEST_H)) printf("NEON verify fails;");
    printf("neon: \t\t %ld us\n", diff_in_us(start, end));
    //printf("sse per iteration: \t\t %lf us\n", (double)diff_in_us(start, end)/sse_iteration);
#endif
#ifdef ARM_PRE
    pthread_t tid[4];
    struct Param args[4];

    args[0].src = src;
    args[0].out = out;
    args[0].w_begin = 0;
    args[0].w_end = 2048;
    args[0].h_begin = 0;
    args[0].h_end = 2048;

    args[1].src = src;
    args[1].out = out;
    args[1].w_begin = 2048;
    args[1].w_end = 4096;
    args[1].h_begin = 0;
    args[1].h_end = 2048;

    args[2].src = src;
    args[2].out = out;
    args[2].w_begin = 0;
    args[2].w_end = 2048;
    args[2].h_begin = 2048;
    args[2].h_end = 4096;

    args[3].src = src;
    args[3].out = out;
    args[3].w_begin = 2048;
    args[3].w_end = 4096;
    args[3].h_begin = 2048;
    args[3].h_end = 4096;

    clock_gettime(CLOCK_REALTIME, &start);

    pthread_create(&tid[0], NULL, (void *) &neon_prefetch_transpose, (void *) &args[0]);
    pthread_create(&tid[1], NULL, (void *) &neon_prefetch_transpose, (void *) &args[1]);
    pthread_create(&tid[2], NULL, (void *) &neon_prefetch_transpose, (void *) &args[2]);
    pthread_create(&tid[3], NULL, (void *) &neon_prefetch_transpose, (void *) &args[3]);
    pthread_join(tid[0], NULL);
    pthread_join(tid[1], NULL);
    pthread_join(tid[2], NULL);
    pthread_join(tid[3], NULL);

    clock_gettime(CLOCK_REALTIME, &end);

    if(!transpose_verify(src, out, TEST_W, TEST_H)) printf("NEON prefetch verify fails;");
    printf("neon_pre: \t\t %ld us\n", diff_in_us(start, end));
    //printf("sse per iteration: \t\t %lf us\n", (double)diff_in_us(start, end)/sse_iteration);
#endif
    //free all used memory
    free(out);
    free(src);
    return 0;
}
