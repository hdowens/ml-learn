#ifndef NEURN_H_
#define NEURN_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NEURN_MALLOC
#include <stdlib.h>
#define NEURN_MALLOC malloc
#endif // NEURN_MALLOC

#ifndef NEURN_ASSERT
#include <assert.h>
#define NEURN_ASSERT assert 
#endif //NEURN_ASSERT

//structure is lightweight because
//only contains 3 64bit integers
//on -> x86_64
//size_t represents the biggest addressable size

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

float rand_float(void);
float sigmoidf(float x);
typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]
#define MAT_PRINT(m) mat_print(m, #m);



Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low, float high);
void mat_fill(Mat m, float n);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_print(Mat a, const char *name);
void mat_sig(Mat m);
//destination is first for these memory copying fncts
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);

typedef struct {
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; //amount of activations is count+1
} NN;


//size_t arch = {2, 2, 1}
// NN nn = nn_alloc(arch, ARRAY_LEN(arch))

NN nn_alloc(size_t *arch, size_t arch_count);

 
#endif //END OF HEADER PART

#ifdef NEURN_IMPLEMENTATION

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}


Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es   = NEURN_MALLOC(sizeof(*m.es) * rows * cols);
    NEURN_ASSERT(m.es != NULL);
    return m;
}

void mat_rand(Mat m, float low, float high)
{
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j < m.cols; ++j){
            MAT_AT(m, i, j) = rand_float() * (high-low) + low;
        }
    }
}

void mat_fill(Mat m, float n)
{
    for(size_t i = 0; i < m.rows; ++i){ 
        for(size_t j = 0; j < m.cols; ++j){
            MAT_AT(m, i, j) = n;
        }
    }
}

Mat mat_row(Mat m, size_t row)
{
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es     = &MAT_AT(m, row, 0)
    };
}


void mat_copy(Mat dst, Mat src)
{
    NEURN_ASSERT(dst.rows == src.rows);
    NEURN_ASSERT(dst.cols == src.cols);
    for(size_t i = 0; i < dst.rows; ++i){
        for(size_t j = 0; j < dst.cols; ++j){
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }

}

void mat_print(Mat m, const char *name)
{
    printf("%s = [\n", name);
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j < m.cols; ++j){
            printf("    %f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

void mat_sig(Mat m)
{
    for(size_t i = 0; i < m.rows; ++i){ 
        for(size_t j = 0; j < m.cols; ++j){
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

void mat_dot(Mat dst, Mat a, Mat b)
{
    //1x(2 . 2)x3 is multipliable 
    //as inner 2 are 2
    //results in a 1x3 matrixc
    NEURN_ASSERT(a.cols == b.rows);
    size_t n = a.cols;
    NEURN_ASSERT(dst.rows == a.rows);
    NEURN_ASSERT(dst.cols == b.cols);

    for(size_t i = 0; i < dst.rows; ++i){
        for(size_t j = 0; j < dst.cols; ++j){
            MAT_AT(dst, i, j) = 0;
            for(size_t k = 0; k < n; ++k){
                //i (k   k) j
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

void mat_sum(Mat dst, Mat a)
{
    NEURN_ASSERT(dst.rows == a.rows);
    NEURN_ASSERT(dst.cols == a.cols);
    for(size_t i = 0; i < dst.rows; ++i){
        for(size_t j = 0; j < dst.cols; ++j){
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

NN nn_alloc(size_t *arch, size_t arch_count)
{
    NEURN_ASSERT(arch_count > 0);

    NN nn;
    nn.count = arch_count - 1;

    nn.ws = NEURN_MALLOC(sizeof(*nn.ws)*nn.count);
    NEURN_ASSERT(nn.ws != NULL);
    nn.bs = NEURN_MALLOC(sizeof(*nn.bs)*nn.count);
    NEURN_ASSERT(nn.bs != NULL);
    nn.as = NEURN_MALLOC(sizeof(*nn.as)*nn.count);
    NEURN_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1, arch[0]);
    for(size_t i = 1; i < arch_count; ++i){
        nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
        nn.bs[i-1] = mat_alloc(1, arch[i]);
        nn.as[i]   = mat_alloc(1, arch[i]);
    }

    return nn;
}

#endif //NEURN_IMPLEMENTATION