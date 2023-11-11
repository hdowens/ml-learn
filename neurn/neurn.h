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
#define MAT_PRINT(m) mat_print(m, #m, 0);

Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low, float high);
void mat_fill(Mat m, float n);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_print(Mat a, const char *name, size_t padding);
void mat_sig(Mat m);
//destination is first for these memory copying fncts
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
typedef struct {
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; //amount of activations is count+1
} NEURN;

#define NEURN_INPUT(nn) (nn).as[0]
#define NEURN_OUTPUT(nn) (nn).as[(nn).count]


NEURN neurn_alloc(size_t *arch, size_t arch_count);
void neurn_print(NEURN nn, const char *name);
#define NEURN_PRINT(nn) neurn_print(nn, #nn); 
void neurn_rand(NEURN nn, float low, float high);
void neurn_forward(NEURN nn);
float neurn_cost(NEURN nn, Mat ti, Mat to);
void neurn_finite_diff(NEURN nn, NEURN g, float eps, Mat ti, Mat to);
void neurn_learn(NEURN nn, NEURN g, float rate);

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

/*
    %*s: This is the format specifier. 
    The asterisk * indicates that the width of the field is specified by an additional 
    argument, and s for string

    (int) padding: This is the additional argument specifying the width of the field. 
    The (int) cast is used to ensure that padding is treated as an integer. 
    The value of padding determines how many spaces will be printed as padding.

    "": An empty string

    This is done to dynamically pad the matrices at runtime

*/

void mat_print(Mat m, const char *name, size_t padding)
{
    printf("%*s%s = [\n", (int) padding, "", name);
    for(size_t i = 0; i < m.rows; ++i){
        printf("%*s    ", (int) padding, "");
        for(size_t j = 0; j < m.cols; ++j){
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n",(int) padding, "");
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

NEURN neurn_alloc(size_t *arch, size_t arch_count)
{
    NEURN_ASSERT(arch_count > 0);

    NEURN nn;
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
        mat_fill(nn.ws[i-1], 0);
        nn.bs[i-1] = mat_alloc(1, arch[i]);
        mat_fill(nn.bs[i-1], 0);
        nn.as[i]   = mat_alloc(1, arch[i]);
        mat_fill(nn.as[i], 0);
    }

    return nn;
}

void neurn_print(NEURN nn, const char *name)
{
    char buf[256];
    printf("%s = [\n", name);
    for(size_t i = 0; i < nn.count; ++i){
        snprintf(buf, sizeof(buf), "ws%zu", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        mat_print(nn.bs[i], buf, 4);
    }
    printf("]\n");
}

void neurn_rand(NEURN nn, float low, float high)
{
    for(size_t i = 0; i < nn.count; ++i){
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void neurn_forward(NEURN nn)
{
    for(size_t i = 0 ; i < nn.count; ++i){
        mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i+1], nn.bs[i]);
        mat_sig(nn.as[i+1]);
    }
}

float neurn_cost(NEURN nn, Mat ti, Mat to)
{
    assert(ti.rows == to.rows);
    assert(to.cols == NEURN_OUTPUT(nn).cols);
    size_t n = ti.rows;
    
    float c = 0;
    for(size_t i = 0; i < n; ++i){
        Mat x = mat_row(ti, i); //expected input
        Mat y = mat_row(to, i); //expected output

        mat_copy(NEURN_INPUT(nn), x);
        neurn_forward(nn);
        NEURN_OUTPUT(nn);
        size_t q = to.cols;
        for(size_t j = 0; j < q; j++){
            float d = MAT_AT(NEURN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            c += d*d;
        }
    }

    return c/n;
}

void neurn_finite_diff(NEURN nn, NEURN g, float eps, Mat ti, Mat to)
{
    float saved;
    float c = neurn_cost(nn, ti, to);

    for(size_t i = 0; i < nn.count; i++){
        for(size_t j = 0; j < nn.ws[i].rows; ++j){
            for(size_t k = 0; k < nn.ws[i].cols; ++k){
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += eps;
                MAT_AT(g.ws[i], j, k) = (neurn_cost(nn, ti, to) - c) / eps;
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }
    
        for(size_t j = 0; j < nn.bs[i].rows; ++j){
            for(size_t k = 0; k < nn.bs[i].cols; ++k){
                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) += eps;
                MAT_AT(g.bs[i], j, k) = (neurn_cost(nn, ti, to) - c) / eps;
                MAT_AT(nn.bs[i], j, k) = saved;
            }
        }
    }
}

void neurn_learn(NEURN nn, NEURN g, float rate)
{
    for(size_t i = 0; i < nn.count; i++){
        for(size_t j = 0; j < nn.ws[i].rows; ++j){
            for(size_t k = 0; k < nn.ws[i].cols; ++k){
                MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(g.ws[i], j, k);
            }
        }
    
        for(size_t j = 0; j < nn.bs[i].rows; ++j){
            for(size_t k = 0; k < nn.bs[i].cols; ++k){
                MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(g.bs[i], j, k);
            }
        }
    }
}


#endif //NEURN_IMPLEMENTATION