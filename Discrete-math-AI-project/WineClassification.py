# [X_Variables]

#       x1 = fixed.acidity
#       x2 = volatile.acidity
#       x3 = citric.acid
#       x4 = residual.sugar
#       x5 = chlorides
#       x6 = free.sulfur.dioxide
#       x7 = total.sulfur.dioxide
#       x8 = density
#       x9 = pH
#       x10 = sulphates
#       x11 = alcohol

# [Derived value]

#       y = quality

# [Algorithm]

#       Linear Regression


import tensorflow as tf 
import numpy as np

# Reading Data
data = np.loadtxt("wineQualityReds.csv", delimiter=",", dtype=np.float32)

# Training Data
x_data = data[ :, 0:-1 ]
y_data = data[ :, [-1] ]

# Testing Data (오차율 측정)
y_test_data = [5, 5, 5, 6, 5, 5, 5, 7, 7, 5]

# Shape은 X_Variables와 Y에 따라 결정
X = tf.placeholder(tf.float32, shape = [None, 11])
Y = tf.placeholder(tf.float32, shape = [None, 1])

# 모든 element의 값이 0인 shape이 [11, 1], [1]인 텐서를 생성
W = tf.Variable(tf.zeros([11, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

# Algorithm: Linear Regression
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Training Operation: Gradient Descent Optimizer
# Optimum Learning Rate: 0.0001
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())                              # 변수 초기화
    for step in range(50001):                                                # Training 50001 times
        _cost, _= sess.run([cost, train], feed_dict={X: x_data, Y: y_data})  # x_data, y_data로 학습 수행
        if step % 500 == 0:                                                  # step 이 0, 500, 1000... 면 Cost Check
            print("Step: ", step, "Cost: ", _cost)

    print("\nTest: ", sess.run(hypothesis, feed_dict={ X: [[7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]] }), "Answer: ", y_test_data[0])      
    print("Test: ", sess.run(hypothesis, feed_dict={ X: [[7.8,0.88,0,2.6,0.098,25,67,0.9968,3.2,0.68,9.8]] }), "Answer: ", y_test_data[1])
    print("Test: ", sess.run(hypothesis, feed_dict={ X: [[7.8,0.76,0.04,2.3,0.092,15,54,0.997,3.26,0.65,9.8]] }), "Answer: ", y_test_data[2])
    print("Test: ", sess.run(hypothesis, feed_dict={ X: [[11.2,0.28,0.56,1.9,0.075,17,60,0.998,3.16,0.58,9.8]] }), "Answer: ", y_test_data[3])
    print("Test: ", sess.run(hypothesis, feed_dict={ X: [[7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]] }), "Answer: ", y_test_data[4])
    
    aa = (abs(y_test_data[0] - sess.run(hypothesis, feed_dict = { X: [[7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]] })) / y_test_data[0])
    ab = (abs(y_test_data[1] - sess.run(hypothesis, feed_dict = { X: [[7.8,0.88,0,2.6,0.098,25,67,0.9968,3.2,0.68,9.8]] })) / y_test_data[1])
    ac = (abs(y_test_data[2] - sess.run(hypothesis, feed_dict = { X: [[7.8,0.76,0.04,2.3,0.092,15,54,0.997,3.26,0.65,9.8]] })) / y_test_data[2])
    ad = (abs(y_test_data[3] - sess.run(hypothesis, feed_dict = { X: [[11.2,0.28,0.56,1.9,0.075,17,60,0.998,3.16,0.58,9.8]] })) / y_test_data[3])
    ae = (abs(y_test_data[4] - sess.run(hypothesis, feed_dict = { X: [[7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]] })) / y_test_data[4])
    af = (abs(y_test_data[5] - sess.run(hypothesis, feed_dict = { X: [[7.4,0.66,0,1.8,0.075,13,40,0.9978,3.51,0.56,9.4]] })) / y_test_data[5])
    ag = (abs(y_test_data[6] - sess.run(hypothesis, feed_dict = { X: [[7.9,0.6,0.06,1.6,0.069,15,59,0.9964,3.3,0.46,9.4]] })) / y_test_data[6])
    ah = (abs(y_test_data[7] - sess.run(hypothesis, feed_dict = { X: [[7.3,0.65,0,1.2,0.065,15,21,0.9946,3.39,0.47,10]] })) / y_test_data[7])
    ai = (abs(y_test_data[8] - sess.run(hypothesis, feed_dict = { X: [[7.8,0.58,0.02,2,0.073,9,18,0.9968,3.36,0.57,9.5]] })) / y_test_data[8])
    aj = (abs(y_test_data[9] - sess.run(hypothesis, feed_dict = { X: [[7.5,0.5,0.36,6.1,0.071,17,102,0.9978,3.35,0.8,10.5]] })) / y_test_data[9])

    print("오차율: ", aa*100, ab*100, ac*100, ad*100, ae*100, af*100, ag*100, ah*100, ai*100, aj*100) # 개별 오차율
    print("오차율 평균: ", (aa+ab+ac+ad+ae+af+ag+ah+ai+aj)*10)                                         # 오차율 평균