---
{"dg-publish":true,"permalink":"/AboutPython/ML&DL/Fraud_Detecting/","tags":["python","딥러닝","머신러닝","DL","ML"],"noteIcon":""}
---


<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


## Credit Card Fraud Detection



약 28만건의 신용카드 거래 데이터 중에서 이상거래(Fraud)를 감지해내는 프로그램



# Module Import



```python
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
```

필요한 라이브러리들을 불러옵니다.




```python
from google.colab import drive
drive.mount('/content/drive')
```

<pre>
Mounted at /content/drive
</pre>
구글 Colab을 사용할 경우, 적절한 작업경로를 지정해줍니다.


# Data Load



```python
df = pd.read_csv("/content/drive/My Drive/FD/creditcard.csv", delimiter=',', dtype=np.float32)
print(df.shape)
df.head()
```

<pre>
(284807, 31)
</pre>
<pre>
   Time        V1        V2        V3        V4        V5        V6        V7  \
0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   
1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   
2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   
3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   
4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   

         V8        V9  ...       V21       V22       V23       V24       V25  \
0  0.098698  0.363787  ... -0.018307  0.277838 -0.110474  0.066928  0.128539   
1  0.085102 -0.255425  ... -0.225775 -0.638672  0.101288 -0.339846  0.167170   
2  0.247676 -1.514654  ...  0.247998  0.771679  0.909412 -0.689281 -0.327642   
3  0.377436 -1.387024  ... -0.108300  0.005274 -0.190321 -1.175575  0.647376   
4 -0.270533  0.817739  ... -0.009431  0.798279 -0.137458  0.141267 -0.206010   

        V26       V27       V28      Amount  Class  
0 -0.189115  0.133558 -0.021053  149.619995    0.0  
1  0.125895 -0.008983  0.014724    2.690000    0.0  
2 -0.139097 -0.055353 -0.059752  378.660004    0.0  
3 -0.221929  0.062723  0.061458  123.500000    0.0  
4  0.502292  0.219422  0.215153   69.989998    0.0  

[5 rows x 31 columns]
</pre>
Pandas로 csv파일을 불러와서 확인해 줍니다. df라는 변수에 데이터가 제대로 저장되었는지 head 명령어를 통해 확인해 줍니다.


# Missing Value Check



```python
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 284807 entries, 0 to 284806
Data columns (total 31 columns):
 #   Column  Non-Null Count   Dtype  
---  ------  --------------   -----  
 0   Time    284807 non-null  float32
 1   V1      284807 non-null  float32
 2   V2      284807 non-null  float32
 3   V3      284807 non-null  float32
 4   V4      284807 non-null  float32
 5   V5      284807 non-null  float32
 6   V6      284807 non-null  float32
 7   V7      284807 non-null  float32
 8   V8      284807 non-null  float32
 9   V9      284807 non-null  float32
 10  V10     284807 non-null  float32
 11  V11     284807 non-null  float32
 12  V12     284807 non-null  float32
 13  V13     284807 non-null  float32
 14  V14     284807 non-null  float32
 15  V15     284807 non-null  float32
 16  V16     284807 non-null  float32
 17  V17     284807 non-null  float32
 18  V18     284807 non-null  float32
 19  V19     284807 non-null  float32
 20  V20     284807 non-null  float32
 21  V21     284807 non-null  float32
 22  V22     284807 non-null  float32
 23  V23     284807 non-null  float32
 24  V24     284807 non-null  float32
 25  V25     284807 non-null  float32
 26  V26     284807 non-null  float32
 27  V27     284807 non-null  float32
 28  V28     284807 non-null  float32
 29  Amount  284807 non-null  float32
 30  Class   284807 non-null  float32
dtypes: float32(31)
memory usage: 33.7 MB
</pre>
불러온 데이터를 조금 더 구체적으로 살펴봅니다. 각 변수에 대해서 missing value가 존재하는지 확인합니다. 만약 missing value가 존재할 경우 count 컬럼에 missing value의 숫자가 기록됩니다. 이 데이터에서는 missing value가 존재하지 않음을 확인할 수 있습니다. 또한 데이터 타입과 메모리 사용량도 확인 가능합니다.



```python
df.isnull().sum() / df.shape[0]
```

<pre>
Time      0.0
V1        0.0
V2        0.0
V3        0.0
V4        0.0
V5        0.0
V6        0.0
V7        0.0
V8        0.0
V9        0.0
V10       0.0
V11       0.0
V12       0.0
V13       0.0
V14       0.0
V15       0.0
V16       0.0
V17       0.0
V18       0.0
V19       0.0
V20       0.0
V21       0.0
V22       0.0
V23       0.0
V24       0.0
V25       0.0
V26       0.0
V27       0.0
V28       0.0
Amount    0.0
Class     0.0
dtype: float64
</pre>
만약 missing value가 존재한다면, 몇개의 missing value가 존재하는지 확인해 줍니다. missing value의 존재를 확인했다면 해당 missing value가 존재하는는 row를 삭제하거나 적절한 값으로 missing value를 채워주도록 합니다. 여기서 적절한 값을 계산하는 방법은 여러가지가 존재할 수 있으며, 가장 간단하게는 평균이나 중간값을 사용할 수 있습니다.


# Correlation Visualize



```python
import seaborn
f, ax = plt.subplots(figsize = (25,15))
seaborn.heatmap(df.corr(), annot=True, linewidths=0.3, fmt="0.2f", ax=ax, cmap="viridis")
plt.show()
```

<pre>
<Figure size 1800x1080 with 2 Axes>
</pre>
Feature들 사이의 상관관계를 확인합니다. Credit card 데이터의 경우에는 이미 한번 PCA를 거친 데이터이기 때문에 Feature들 사이의 상관관계가 매우 낮게 나옵니다. 그러나 만약 Feature들 간의 상관관계가 높게 나오는 상황을 마주하게 된다면 VIF를 통해서 다중공선성을 확인해주는것이 좋습니다.


# EDA

데이터의 전체적인 구조를 살펴봅니다.



```python
print(df["Class"].value_counts())
count_classes = pd.value_counts(df["Class"], sort=True)
count_classes.head()
print()

print(df["Class"].value_counts(normalize=True))
```

<pre>
0.0    284315
1.0       492
Name: Class, dtype: int64

0.0    0.998273
1.0    0.001727
Name: Class, dtype: float64
</pre>
0.0 : 정상거래

1.0 : 이상거래



정상거래는 28만 4315건, 이상거래는 492건이 존재합니다.

전체 데이터셋에서 오직 0.0017%가 이상거래이므로 심각한 imbalance가 존재하는 dataset임을 파악할 수 있습니다. 그러므로 일반적인 상황에서 주로 사용되는 Accuracy를 사용해서 모델의 Performance를 측정하기 어렵다는 결론이 도출됩니다. 따라서 모델을 설계할 때 Precision, Recall, 그리고 F1 Score를 사용하도록 디자인 해줍니다.



```python
labels = ["Normal", "Fraud"]
count_classes.plot(kind="bar")
plt.xticks(range(2), labels)
plt.title("Transaction Class Distribution")
```

<pre>
Text(0.5, 1.0, 'Transaction Class Distribution')
</pre>
<pre>
<Figure size 432x288 with 1 Axes>
</pre>
Data imabalance가 얼마나 심한지 시각화를 통해서 확인합니다.



```python
df.drop(columns = "Time", inplace=True)
df.head()
```

<pre>
         V1        V2        V3        V4        V5        V6        V7  \
0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   
1  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   
2 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   
3 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   
4 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   

         V8        V9       V10  ...       V21       V22       V23       V24  \
0  0.098698  0.363787  0.090794  ... -0.018307  0.277838 -0.110474  0.066928   
1  0.085102 -0.255425 -0.166974  ... -0.225775 -0.638672  0.101288 -0.339846   
2  0.247676 -1.514654  0.207643  ...  0.247998  0.771679  0.909412 -0.689281   
3  0.377436 -1.387024 -0.054952  ... -0.108300  0.005274 -0.190321 -1.175575   
4 -0.270533  0.817739  0.753074  ... -0.009431  0.798279 -0.137458  0.141267   

        V25       V26       V27       V28      Amount  Class  
0  0.128539 -0.189115  0.133558 -0.021053  149.619995    0.0  
1  0.167170  0.125895 -0.008983  0.014724    2.690000    0.0  
2 -0.327642 -0.139097 -0.055353 -0.059752  378.660004    0.0  
3  0.647376 -0.221929  0.062723  0.061458  123.500000    0.0  
4 -0.206010  0.502292  0.219422  0.215153   69.989998    0.0  

[5 rows x 30 columns]
</pre>
Time 이라는 Feature는 단순히 신용카드 거래가 이루어진 순서를 기록한 것이기 때문에 분석에 크게 도움이 되지 않습니다. \

그러므로 데이터 테이블에서 배제시켜 줍니다.



```python
x_data = df.iloc[:,0:-1].values
y_data = df.iloc[:,[-1]].values

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

print(x_data.shape, y_data.shape)
```

<pre>
(284807, 29) (284807, 1)
</pre>
Feature들로 이루어진 x_data와, label을 나타내는 y_data로 원본 데이터를 분리시켜 줍니다. \

분리된 데이터는 앞으로 계산을 해주어야 하기 때문에 동일한 타입의 실수형 데이터로 저장합니다.


# Data Preprocessing - Normalize



```python
scaler = MinMaxScaler()
x_data = scaler.fit_transform(x_data)
print(x_data)
```

<pre>
[[9.3519241e-01 7.6649040e-01 8.8136494e-01 ... 4.1897613e-01
  3.1269664e-01 5.8237929e-03]
 [9.7854203e-01 7.7006662e-01 8.4029853e-01 ... 4.1634512e-01
  3.1342265e-01 1.0470528e-04]
 [9.3521708e-01 7.5311762e-01 8.6814088e-01 ... 4.1548926e-01
  3.1191131e-01 1.4738923e-02]
 ...
 [9.9090487e-01 7.6407969e-01 7.8110206e-01 ... 4.1659316e-01
  3.1258485e-01 2.6421540e-03]
 [9.5420909e-01 7.7285570e-01 8.4958714e-01 ... 4.1851953e-01
  3.1524515e-01 3.8923896e-04]
 [9.4923186e-01 7.6525640e-01 8.4960151e-01 ... 4.1646636e-01
  3.1340083e-01 8.4464857e-03]]
</pre>
데이터의 Scale에 따라서 결과가 왜곡되는 것을 방지하기 위해서 MinMaxScaler를 통해서 데이터가 0~1 사이의 값을 지니도록 조정해 줍니다.


# Logistic Regression


## Parameter Initilization



```python
tf.random.set_seed(2022)

W = tf.Variable(tf.random.normal([29,1], mean=0.0))
b = tf.Variable(tf.random.normal([1], mean=0.0))
```

W: 앞에는 input dimension, 뒤에는 output dimension으로 설정해야 합니다.\

b: output dimension과 동일하게 설정합니다.


## Hypothesis Define



```python
def hypothesis(x):
  z = tf.matmul(x,W) + b
  sigmoid = 1 / (1 + tf.exp(-z))
  return sigmoid
```

Feature와 Label 사이의 관계를 정의하는 부분입니다.\

여기서는 이진분류 문제에 주로 사용되는 Logistic Regression을 hypothesis로 설정해 주었습니다.


## Cost Function Define



```python
def cost_function(H, Y):
  cost = -tf.reduce_mean( Y*tf.math.log(H) + (1-Y)*tf.math.log(1-H))
  return cost
```

Cost Function 혹은 Loss Function으로 불립니다. \

여기서는 마찬가지로 이진분류에서 주로 사용되는 Cross Entropy를 cost function으로 설정해 주었습니다.


## Metric Define



```python
def accuracy(hypo, label):
  predicted = tf.cast( hypo > 0.5, dtype = tf.float32)
  accuracyd = tf.reduce_mean(tf.cast(tf.equal(predicted, label), dtype=tf.float32))
  return accuracyd
```

실제 모델의 퍼포먼스를 측정하는 기준을 설정합니다. \

일반적으로 가장 많이 쓰이는 Accuracy를 우선은 설정해 줍니다.


## Hyper-parameter Setting



```python
learning_rate = 0.0001
optimizer = tf.optimizers.SGD(learning_rate)
```

학습을 진행하기 위해서 우리가 지정해 주어야 하는 변수를 Hyper-parameter라고 합니다. \

대표적인 하이퍼 파라미터는 learning rate, optimizer, epochs, actvation function, number of layer, number of neuron 같은 것들이 존재합니다. \

지금은 딥러닝 모델을 사용하는 것이 아니기 때문에 Logistic Regression 학습에 필요한 learning rate와 optimizer만 설정해 주었습니다.


## Training



```python
for step in range(2022):
  with tf.GradientTape() as g:
    pred = hypothesis(x_data)
    cost = cost_function(pred, y_data)

    gradients = g.gradient(cost, [W,b])

  optimizer.apply_gradients(zip(gradients, [W,b]))

  if step % 200 == 0:
    print(f"step: {step}, loss: {cost.numpy()}")

w_hat = W.numpy()
b_hat = b.numpy()
```

<pre>
step: 0, loss: 0.026920180767774582
step: 200, loss: 0.02684672921895981
step: 400, loss: 0.02677365578711033
step: 600, loss: 0.02670133300125599
step: 800, loss: 0.026629552245140076
step: 1000, loss: 0.02655830793082714
step: 1200, loss: 0.026487411931157112
step: 1400, loss: 0.0264168418943882
step: 1600, loss: 0.02634689025580883
step: 1800, loss: 0.026278041303157806
step: 2000, loss: 0.026209723204374313
</pre>
Backpropagation을 통해서 최적의 W와 b를 계속해서 업데이트 해줍니다. \

최종적으로 업데이트된 값을 w_hat과 b_hat에 저장합니다.


## Performance



```python
acc = accuracy(hypothesis(x_data), y_data).numpy()
print(f"Accuracy: {acc}")
```

<pre>
Accuracy: 0.9982725381851196
</pre>
99.83%의 상당히 높은 Accuracy를 보입니다. \

그러나 위에서 살펴본 바에 의하면 dataset이 매우 imbalance 했으므로, Accuracy는 모델을 평가하는 적절한 metric이 될 수 없습니다. \

그러므로 Precision과 Recall의 측면에서 모델을 다시한번 평가해 봅니다.


## Precision, Recall, F1 Score



```python
# metric
from sklearn.metrics import accuracy_score, classification_report, f1_score
```


```python
predicted_x = tf.cast( hypothesis(x_data) > 0.5, dtype = tf.float32)
```


```python
print("Accuracy = ", accuracy_score(predicted_x, y_data))
print("Report = \n", classification_report(predicted_x, y_data))
```

<pre>
Accuracy =  0.9982725143693799
</pre>
<pre>
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
</pre>
<pre>
Report = 
               precision    recall  f1-score   support

         0.0       1.00      1.00      1.00    284807
         1.0       0.00      0.00      0.00         0

    accuracy                           1.00    284807
   macro avg       0.50      0.50      0.50    284807
weighted avg       1.00      1.00      1.00    284807

</pre>
<pre>
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
</pre>
대안적인 metric의 관점에서 Logisic Regression 모델을 평가해 봤을 때, Fraud transaction을 거의 잡아내지 못하는 것을 볼 수 있습니다. \

그러므로 다른 모델을 통해서 접근할 필요가 있습니다. \





* Precision: 모델이 True라고 분류한 것 중에서 실제 True가 차지하는 비중

* Recall: 실제 True 중에서 모델이 True라고 분류한 데이터의 비중

* F1 score: Precision과 Recall의 조화평균


# Neural Networks

인공신경망 모델을 통해서 Fraud Detecting을 시도해 봅니다.



```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from keras import optimizers, metrics, callbacks
```

## Dataset Split



```python
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2,  random_state=22)
```

학습에 사용할 training set과 퍼포먼스 측정에 사용할 test set을 8:2로 나누어 줍니다.



```python
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.2, random_state=22)
```

EarlyStopping(=callbacks) 사용을 위해서 validation set을 만들어 주어야 합니다.\

기존의 training set 데이터를 또다시 8:2로 나누어 training set과 validation set으로 나누어 줍니다.



```python
print(x_train.shape, y_train.shape)
print(x_validate.shape, y_validate.shape)
print(x_test.shape, y_test.shape)
```

<pre>
(182276, 29) (182276, 1)
(45569, 29) (45569, 1)
(56962, 29) (56962, 1)
</pre>
## Neural Network Design



```python
model = Sequential([
    Dense(256, activation='relu', input_shape=(x_train.shape[-1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(10, activation='softmax'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid'),
])

model.summary()
```

<pre>
Model: "sequential_7"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_49 (Dense)            (None, 256)               7680      
                                                                 
 batch_normalization_42 (Bat  (None, 256)              1024      
 chNormalization)                                                
                                                                 
 dropout_42 (Dropout)        (None, 256)               0         
                                                                 
 dense_50 (Dense)            (None, 128)               32896     
                                                                 
 batch_normalization_43 (Bat  (None, 128)              512       
 chNormalization)                                                
                                                                 
 dropout_43 (Dropout)        (None, 128)               0         
                                                                 
 dense_51 (Dense)            (None, 64)                8256      
                                                                 
 batch_normalization_44 (Bat  (None, 64)               256       
 chNormalization)                                                
                                                                 
 dropout_44 (Dropout)        (None, 64)                0         
                                                                 
 dense_52 (Dense)            (None, 32)                2080      
                                                                 
 batch_normalization_45 (Bat  (None, 32)               128       
 chNormalization)                                                
                                                                 
 dropout_45 (Dropout)        (None, 32)                0         
                                                                 
 dense_53 (Dense)            (None, 16)                528       
                                                                 
 batch_normalization_46 (Bat  (None, 16)               64        
 chNormalization)                                                
                                                                 
 dropout_46 (Dropout)        (None, 16)                0         
                                                                 
 dense_54 (Dense)            (None, 10)                170       
                                                                 
 batch_normalization_47 (Bat  (None, 10)               40        
 chNormalization)                                                
                                                                 
 dropout_47 (Dropout)        (None, 10)                0         
                                                                 
 dense_55 (Dense)            (None, 1)                 11        
                                                                 
=================================================================
Total params: 53,645
Trainable params: 52,633
Non-trainable params: 1,012
_________________________________________________________________
</pre>
자신만의 적절한 ANN 모델을 만들어 줍니다. \

\

Input Layer에서는 항상 input data의 dimension에 따라서 적절한 input_shape을 지정해 주어야합니다. HIdden Layer 사이의 activation function으로는 ReLU를 기본적으로 사용하며, 이진분류 문제이므로 Output Layer에서는 activation function을 꼭 sigmoid로 지정해 주어야 합니다. \

\

Hidden Layer마다 Drop out을 적용해서 Overfitting을 방지하도록 합니다. Batch Normalization도 적용해 주면서 학습이 보다 빠르고 안정적으로 진행되게끔 유도합니다.

\

\

이 모델에서는 Hidden Layer를 지날수록 점점 neuron의 숫자를 줄게하여 정보가 압축되도록 설계했고, Ouput Layer 직전의 Hidden Layer에서는 activation function으로 softmax를 사용했습니다.


## Model compile



```python
model.compile(optimizer=optimizers.Adam(1e-4),
              loss = "binary_crossentropy",
              metrics =  [metrics.Recall(name="recall"),
                          metrics.Precision(name="precision")])
```


```python
callbacks = [callbacks.ModelCheckpoint('epcoh.h5')]
```

모델을 학습하는데 필요한 하이퍼 파라미터들을 지정해 줍니다. \

\

Optimizer는 일반적으로 가장 성능이 좋은 Adam을 사용해줍니다. \

\

Adam에 default 값으로 지정된 learning rate는 1e-3 입니다. 그러나 학습을 진행해 본 결과 learning rate이 너무 커서 학습이 불안정하게 진행되는 것이 포착되었습니다. 그러므로 learning rate을 1e-4로 더 낮추어서 적용해 줍니다. \

\

loss function으로는 이진분류에 적합한 binary crossentropy를 사용합니다.\

\

최종적인 성능평가의 metric으로 precision과 recall을 지정해 줍니다. \

\

최적의 모델을 저장해줍니다. (callbacks)


# Training


batch size: 65536\

epochs: 10000\

EarlyStopping: Yes




```python
history = model.fit(x_train, y_train,
                    validation_data = (x_validate, y_validate),
                    batch_size = 65536,
                    epochs = 10000,
                    callbacks = callbacks
                    )
```


```python
score = model.evaluate(x_test, y_test)
print(score)
```

<pre>
1781/1781 [==============================] - 8s 5ms/step - loss: 0.0035 - recall: 0.8602 - precision: 0.9091
[0.0035365174990147352, 0.8602150678634644, 0.9090909361839294]
</pre>
# Training Visualization



```python
plt.figure(figsize = (30,15))

plt.subplot(3,1,1)
plt.plot(history.history["loss"], label = "Loss")
plt.plot(history.history["val_loss"], label = "Val_Loss")
plt.title("Training Loss & Validate Loss")
plt.legend()

plt.subplot(3,1,2)
plt.plot(history.history["precision"], label = "Precision")
plt.plot(history.history["val_precision"], label = "Val_Precision")
plt.title("Training Precision & Validate Precision")
plt.legend()

plt.subplot(3,1,3)
plt.plot(history.history["recall"], label = "Recall")
plt.plot(history.history["val_recall"], label = "Val_Recall")
plt.title("Training Recall & Validate Recall")
plt.legend()
```

<pre>
<matplotlib.legend.Legend at 0x7f327d67a9d0>
</pre>
<pre>
<Figure size 2160x1080 with 3 Axes>
</pre>
# ANN Performance



```python
Ann_predict = tf.cast( model.predict(x_test) > 0.5, dtype = tf.float32)
print("Test Report = \n", classification_report(Ann_predict, y_test))
```

<pre>
1781/1781 [==============================] - 4s 2ms/step
Test Report = 
               precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     56874
         1.0       0.86      0.91      0.88        88

    accuracy                           1.00     56962
   macro avg       0.93      0.95      0.94     56962
weighted avg       1.00      1.00      1.00     56962

</pre>
# Other Machine Learning Method



```python
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state=22)
```

다른 Machine Learning Method는 EarlyStopping을 사용하지 않으므로 validation set을 만들지 않고, 모두 training data로 사용한다.


# 1) XGboost



```python
from xgboost import XGBClassifier

Xgboost_model = XGBClassifier()
Xgboost_model.fit(x_train, y_train, eval_metric='aucpr')

Xg_predict = tf.cast( Xgboost_model.predict(x_test) > 0.5, dtype = tf.float32)
print("Test Report = \n", classification_report(Xg_predict, y_test))
```

<pre>
/usr/local/lib/python3.7/dist-packages/sklearn/preprocessing/_label.py:98: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/usr/local/lib/python3.7/dist-packages/sklearn/preprocessing/_label.py:133: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
</pre>
<pre>
Test Report = 
               precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     56879
         1.0       0.84      0.94      0.89        83

    accuracy                           1.00     56962
   macro avg       0.92      0.97      0.94     56962
weighted avg       1.00      1.00      1.00     56962

</pre>
# 2) RandomForest



```python
from sklearn.ensemble import RandomForestClassifier

RandomForest_model = RandomForestClassifier(n_estimators=100, oob_score=False)
RandomForest_model.fit(x_train, y_train)

RF_predict = tf.cast( RandomForest_model.predict(x_test) > 0.5, dtype = tf.float32)
print("Test Report = \n", classification_report(RF_predict, y_test))
```

<pre>
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:4: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  after removing the cwd from sys.path.
</pre>
<pre>
Test Report = 
               precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     56882
         1.0       0.82      0.95      0.88        80

    accuracy                           1.00     56962
   macro avg       0.91      0.97      0.94     56962
weighted avg       1.00      1.00      1.00     56962

</pre>
# 3) Light Gradient Boosting



```python
from lightgbm import LGBMClassifier

LGBM_model = LGBMClassifier()
LGBM_model.fit(x_train, y_train, eval_metric='aucpr')

LGBM_predict = tf.cast( LGBM_model.predict(x_test) > 0.5, dtype = tf.float32)
print("Test Report = \n", classification_report(LGBM_predict, y_test))
```

<pre>
/usr/local/lib/python3.7/dist-packages/sklearn/preprocessing/_label.py:98: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/usr/local/lib/python3.7/dist-packages/sklearn/preprocessing/_label.py:133: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
</pre>
<pre>
Test Report = 
               precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     56838
         1.0       0.10      0.07      0.08       124

    accuracy                           1.00     56962
   macro avg       0.55      0.54      0.54     56962
weighted avg       1.00      1.00      1.00     56962

</pre>
# 4) Support Vector Machine



```python
from sklearn.svm import SVC

SVC_model = SVC(kernel="rbf")
SVC_model.fit(x_train, y_train)

SVC_predict = SVC_model.predict(x_test)
print("Test Report = \n", classification_report(SVC_predict, y_test))
```

<pre>
/usr/local/lib/python3.7/dist-packages/sklearn/utils/validation.py:993: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
</pre>
<pre>
Test Report = 
               precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     56872
         1.0       0.82      0.84      0.83        90

    accuracy                           1.00     56962
   macro avg       0.91      0.92      0.92     56962
weighted avg       1.00      1.00      1.00     56962

</pre>
# Model Comparison



```python
F1_dict={}
F1_dict["ANN"] = {"Test" : f1_score(Ann_predict, y_test)}
F1_dict["XGboost"] = {"Test" : f1_score(Xg_predict, y_test)}
F1_dict["RandomForest"] = {"Test" : f1_score(RF_predict, y_test)}
F1_dict["LGBM"] = {"Test" : f1_score(LGBM_predict, y_test)}
F1_dict["SVM"] = {"Test" : f1_score(SVC_predict, y_test)}
```


```python
F1_df = pd.DataFrame(F1_dict)
F1_df.plot(kind='barh', figsize=(15, 8))
```

<pre>
<matplotlib.axes._subplots.AxesSubplot at 0x7f32a07bb2d0>
</pre>
<pre>
<Figure size 1080x576 with 1 Axes>
</pre>

```python
model.save('fraud.h5')
model = tf.keras.models.load_model
```
