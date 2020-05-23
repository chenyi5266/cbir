def compute_average_precision_score(test_codes, test_labels, learned_codes, y_train, n_samples):
	# 对于每个 n_samples 存储相应的标签和距离
    out_labels = []
    out_distances = []
    
	# 对每个图像特征，计算离训练数据集中最近的那些图像
    for i in range(len(test_codes)):
        distances = []
		# 从训练数据集中，计算每个特征距离
        for code in learned_codes:
            distance = np.linalg.norm(code - test_codes[i])
            distances.append(distance)
        
		# 从训练数据集中，存储计算后的距离和响应的标签
        distances = np.array(distances)
        
		# 评分函数需要替换相似的标签
        labels = np.copy(y_train).astype('float32')
        labels[labels != test_labels[i]] = -1
        labels[labels == test_labels[i]] = 1
        labels[labels == -1] = 0
        distance_with_labels = np.stack((distances, labels), axis=-1)
        sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 0].argsort()]
        
		# 距离在0-28之间、距离越近评分越高
        sorted_distances = 28 - sorted_distance_with_labels[:, 0]
        sorted_labels = sorted_distance_with_labels[:, 1]
        
		# 从取回的图像中仅保存最近的元素
        out_distances.append(sorted_distances[:n_samples])
        out_labels.append(sorted_labels[:n_samples])

    out_labels = np.array(out_labels)
    out_labels_file_name = 'computed_data/out_labels_{}'.format(n_samples)
    np.save(out_labels_file_name, out_labels)

    out_distances_file_name = 'computed_data/out_distances_{}'.format(n_samples)
    out_distances = np.array(out_distances)
    np.save(out_distances_file_name, out_distances)
    
	# 对第一个图片计算模型评分
    score = label_ranking_average_precision_score(out_labels, out_distances)
    scores.append(score)
    return score