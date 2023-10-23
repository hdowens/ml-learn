### Machine Learning
Essentially another paradigm of programming. Traditionally, you write code for what you want to do. In ML, you dont write the code for the wanted outcome. You write code to produce a model to do the thing you want to do. "Automatic mathematical modelling" 

1. conduct a bunch of experiments
2. you put the data of the experiments into the computer
3. computer does the automatic mathametical modelling (ML!)
4. you got the model, do whatever the fuck you want to do

we have the data, we have the expected behaviour and we tweak the model until it is to our liking, and see which model will feed the specific data

you can model _anything_
 

actual: 0.000000, expected: 0.000000
actual: 7.107210, expected: 2.000000
actual: 14.214420, expected: 4.000000
actual: 21.321630, expected: 6.000000
actual: 28.428841, expected: 8.000000

this model is shit - literally pile of steamy shite. The point is, is that this is "ml". We have a dataset, we have a parameter which we are trying to "model" the data, and we are comparing the outcomes

comparison metrics - the reason we have the square is 1. so it doesnt matter if its negative really, and 2. more crucially, error is amplified

```
    float result = 0.0f;
    for(size_t i = 0; i < train_count; i++){
        float x = train[i][0];
        float y = x * w;
        float error = y - train[i][1];
        result += error*error;
    }

```

this type of thing is called the cost/loss function

our cost function is essentially a mathematical function. 
Imagine a parabola represents the cost function- at any time we are at a given point. Essentially, if we take the derivative of that function, it will tell us in which direction the function grows. We need to move in the opposite grows, which means we approach the minimum of the cost function!

learning rate is the rate at which the derivative estimation should change - if its too big the model is going to overshoot, right?

