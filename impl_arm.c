#ifndef TRANSPOSE_IMPL
#define TRANSPOSE_IMPL
long neon_iteration;

struct Param {
    int *src;
    int *out;
    int w_begin;
    int w_end;
    int h_begin;
    int h_end;
};

int transpose_verify(int *test_src, int *test_dest, int w, int h);
void naive_transpose(int *src, int *dst, int w, int h);
void neon_transpose(int *src, int *dst, int w, int h);
void neon_prefetch_transpose(void *args);

//verify
int transpose_verify(int *test_src, int *test_dest, int w, int h)
{
    int *expected  = (int *) malloc(sizeof(int) * w * h);
    naive_transpose(test_src, expected, w, h);
    if(memcmp(test_dest, expected, w*h*sizeof(int)) != 0) {
        free(expected);
        return 1;
    } else {
        free(expected);
        return 0;
    }
}
void naive_transpose(int *src, int *dst, int w, int h)
{
    //naive_iteration = 0;
    for (int x = 0; x < w; x++)
        for (int y = 0; y < h; y++) {
            *(dst + x * h + y) = *(src + y * w + x);
        } //naive_iteration +=1;}
}


void neon_transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x += 4) {
        for(int y = 0; y < h; y += 4) {
            int32x4_t I0 = vld1q_s32((int32_t *)(src + (y + 0) * w + x));
            int32x4_t I1 = vld1q_s32((int32_t *)(src + (y + 1) * w + x));
            int32x4_t I2 = vld1q_s32((int32_t *)(src + (y + 2) * w + x));
            int32x4_t I3 = vld1q_s32((int32_t *)(src + (y + 3) * w + x));

            vzipq_s32(I0, I1); //I0: T0, I1:T2
            vzipq_s32(I2, I3); //I2: T1, I3:T3

            int32x4_t T0 = vcombine_s32(vget_low_s32(I0), vget_low_s32(I1));//vcombine_s32(low,high)
            int32x4_t T1 = vcombine_s32(vget_high_s32(I0), vget_high_s32(I1));
            int32x4_t T2 = vcombine_s32(vget_low_s32(I2), vget_low_s32(I3));
            int32x4_t T3 = vcombine_s32(vget_high_s32(I2), vget_high_s32(I3));

            vst1q_s32((int32_t *)(dst + ((x + 0) * h) + y), T0);
            vst1q_s32((int32_t *)(dst + ((x + 1) * h) + y), T1);
            vst1q_s32((int32_t *)(dst + ((x + 2) * h) + y), T2);
            vst1q_s32((int32_t *)(dst + ((x + 3) * h) + y), T3);
        }
    }
}


void neon_prefetch_transpose(void *args)
{
    struct Param *params = (struct Param *) args;
    int *src = params->src;
    int *dst = params->out;
    int w = params->w_end - params->w_begin;
    int h = params->h_end - params->h_begin;
    int h2 = params->h_begin + h / 2;

    for (int x = params->w_begin; x < params->w_end; x += 4) {
        for(int y = params->h_begin; y < h2; y += 8) {
#define PFDIST  8
            __builtin_prefetch(src+(y + PFDIST + 0) *w + x);
            __builtin_prefetch(src+(y + PFDIST + 1) *w + x);
            __builtin_prefetch(src+(y + PFDIST + 2) *w + x);
            __builtin_prefetch(src+(y + PFDIST + 3) *w + x);
            __builtin_prefetch(src+(y + PFDIST + 4) *w + x);
            __builtin_prefetch(src+(y + PFDIST + 5) *w + x);
            __builtin_prefetch(src+(y + PFDIST + 6) *w + x);
            __builtin_prefetch(src+(y + PFDIST + 7) *w + x);

            int32x4_t I0 = vld1q_s32((int32_t *)(src + (y + 0) * w + x));
            int32x4_t I1 = vld1q_s32((int32_t *)(src + (y + 1) * w + x));
            int32x4_t I2 = vld1q_s32((int32_t *)(src + (y + 2) * w + x));
            int32x4_t I3 = vld1q_s32((int32_t *)(src + (y + 3) * w + x));
            int32x4_t I4 = vld1q_s32((int32_t *)(src + (y + 4) * w + x));
            int32x4_t I5 = vld1q_s32((int32_t *)(src + (y + 5) * w + x));
            int32x4_t I6 = vld1q_s32((int32_t *)(src + (y + 6) * w + x));
            int32x4_t I7 = vld1q_s32((int32_t *)(src + (y + 7) * w + x));

            vzipq_s32(I0, I1); //I0: T0, I1:T2
            vzipq_s32(I2, I3); //I2: T1, I3:T3
            vzipq_s32(I4, I5); //I0: T0, I1:T2
            vzipq_s32(I6, I7); //I2: T1, I3:T3

            int32x4_t T0 = vcombine_s32(vget_low_s32(I0), vget_low_s32(I1));//vcombine_s32(low,high)
            int32x4_t T1 = vcombine_s32(vget_high_s32(I0), vget_high_s32(I1));
            int32x4_t T2 = vcombine_s32(vget_low_s32(I2), vget_low_s32(I3));
            int32x4_t T3 = vcombine_s32(vget_high_s32(I2), vget_high_s32(I3));
            int32x4_t T4 = vcombine_s32(vget_low_s32(I4), vget_low_s32(I5));//vcombine_s32(low,high)
            int32x4_t T5 = vcombine_s32(vget_high_s32(I4), vget_high_s32(I5));
            int32x4_t T6 = vcombine_s32(vget_low_s32(I6), vget_low_s32(I7));
            int32x4_t T7 = vcombine_s32(vget_high_s32(I6), vget_high_s32(I7));

            vst1q_s32((int32_t *)(dst + ((x + 0) * h) + y), T0);
            vst1q_s32((int32_t *)(dst + ((x + 1) * h) + y), T1);
            vst1q_s32((int32_t *)(dst + ((x + 2) * h) + y), T2);
            vst1q_s32((int32_t *)(dst + ((x + 3) * h) + y), T3);
            vst1q_s32((int32_t *)(dst + ((x + 4) * h) + y), T4);
            vst1q_s32((int32_t *)(dst + ((x + 5) * h) + y), T5);
            vst1q_s32((int32_t *)(dst + ((x + 6) * h) + y), T6);
            vst1q_s32((int32_t *)(dst + ((x + 7) * h) + y), T7);
        }
    }

    pthread_exit(0);
}
#endif /* TRANSPOSE_IMPL */
