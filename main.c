#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//first number is input, second is what we expect
float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8}
};

#define train_count (sizeof(train)/sizeof(train[0]))

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}

//just accepting the model as a parameter
//in this case it is just one parameter
float cost(float w)
{
    float result = 0.0f;
    for(size_t i = 0; i < train_count; i++){
        float x = train[i][0];
        float y = x * w;
        float error = y - train[i][1];
        result += error*error;
    }

    //closer we have to 0 is "theoretically" good, but there is the problem of overfitting
    result /= train_count;
    return result;
}

int main()
{
    //srand(time(0));
    srand(69);
    //all we know is y = x*w , x is input , w is a "parameter" of our model
    float w = rand_float()*10.0f;

    //eps here is just generally trying to shift the parameter
    float eps = 1e-3;
    float rate = 1e-3;
    for (size_t i = 0; i < 500; ++i){
        float dcost = (cost(w + eps) - cost(w) )/ eps;  
        w -= rate*dcost; //well known method called finite difference
        printf("%f\n", cost(w));
    }
    printf("-----------------------------------\n");
    printf("%f\n", w);

    return 0;
}