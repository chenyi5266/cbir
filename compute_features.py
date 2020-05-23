import numpy as np
from keras.models import Model
from keras.datasets import mnist
from keras.models import load_model
from sklearn.metrics import label_ranking_average_precision_score

# 加载数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 加载训练的模型
autoencoder = load_model('autoencoder.h5')

# 从训练模型中获取编译器层
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)

#　存储计算的评分
scores = []

# 仅取1000个图片进行测试
n_test_samples = 1000

# 每次存储前10个图片
n_train_samples = [10, 50, 100, 200, 300, 400, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                   20000, 30000, 40000, 50000, 60000]


def test_model(n_test_samples, n_train_samples):
	# 计算训练集特征
    learned_codes = encoder.predict(x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    learned_codes = learned_codes.reshape(learned_codes.shape[0], learned_codes.shape[1] * learned_codes.shape[2] * learned_codes.shape[3])
    
	# 计算图片的特征
    test_codes = encoder.predict(x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    test_codes = test_codes.reshape(test_codes.shape[0], test_codes.shape[1] * test_codes.shape[2] * test_codes.shape[3])
    
	# 仅保持测试训练集中的部分图像
    indexes = np.arange(len(y_test))
    np.random.shuffle(indexes)
    indexes = indexes[:n_test_samples]
    
	# 计算评分
    score = compute_average_precision_score(test_codes[indexes], y_test[indexes], learned_codes, y_train, n_train_samples)

for n_train_sample in n_train_samples:
    test_model(n_test_samples, n_train_sample)
 
# 将计算的评分保存到文件中
np.save('computed_data/scores', np.array(scores))