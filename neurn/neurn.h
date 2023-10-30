#ifndef NEURN_H_
#define NEURN_H_

#include <stddef.h>
#include <stdio.h>

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
typedef struct {
    size_t rows;
    size_t cols;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).cols + (j)]

float rand_float(void);


Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low, float high);
void mat_fill(Mat m, float n);
void mat_print(Mat a);
//destination is first for these memory copying fncts
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);

 
#endif //END OF HEADER PART

#ifdef NEURN_IMPLEMENTATION

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}


Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.es   = NEURN_MALLOC(sizeof(*m.es) * rows * cols);
    NEURN_ASSERT(m.es != NULL);
    return m;
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

void mat_fill(Mat m, float n)
{
    for(size_t i = 0; i < m.rows; ++i){ 
        for(size_t j = 0; j < m.cols; ++j){
            MAT_AT(m, i, j) = n;
        }
    }
}

void mat_print(Mat m)
{
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j < m.cols; ++j){
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
}

void mat_rand(Mat m, float low, float high)
{
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j < m.cols; ++j){
            MAT_AT(m, i, j) = rand_float() * (high-low) + low;
        }
    }
}

#endif //NEURN_IMPLEMENTATION