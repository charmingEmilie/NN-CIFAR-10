import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 数据加载与预处理
def load_cifar10(path):
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    X_train = []
    y_train = []
    for i in range(1,6):
        data = unpickle(os.path.join(path, f'data_batch_{i}'))
        X_train.append(data[b'data'])
        y_train.extend(data[b'labels'])
    
    test_data = unpickle(os.path.join(path, 'test_batch'))
    X_test = test_data[b'data']
    y_test = test_data[b'labels']
    
    X_train = np.vstack(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # 归一化
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    
    # 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# 神经网络模型
class ThreeLayerNet:
    '''前向传播、损失计算、反向传播、参数更新'''
    def __init__(self, input_size, hidden_size, output_size, activation='relu', reg=0):
        # 初始化第一层权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros(hidden_size)
        # 初始化第二层权重和偏置
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
        self.b2 = np.zeros(output_size)
        # 激活函数类型
        self.activation = activation
        # 正则化系数
        self.reg = reg
        
    def forward(self, X):
        '''
        X：形状为 (N, D)，其中 N 是样本数量，D 是每个样本的特征数量。
        self.W1：形状为 (D, H)，D 是输入特征数，H 是隐藏层神经元数量。
        self.b1：形状为 (H,)。
        self.W2：形状为 (H, C)，H 是隐藏层神经元数量，C 是输出类别数量。
        self.b2：形状为 (C,)。

        使用 CIFAR - 10 数据集的情况下，每张图像是 32x32 像素的 RGB 图像，这意味着每张图像有 32x32x3 = 3072 个像素值。因此，输入特征的数量 D 是 3072
        '''
        # 第一层线性变换
        self.z1 = X.dot(self.W1) + self.b1
        # 根据激活函数类型选择激活方式
        if self.activation == 'relu':
            self.a1 = np.maximum(0, self.z1)
        elif self.activation == 'sigmoid':
            self.a1 = 1/(1+np.exp(-self.z1))
        # 第二层线性变换
        self.z2 = self.a1.dot(self.W2) + self.b2
        # 计算softmax概率(将模型的输出转换为概率分布)
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        # 将 exp_scores 中的每个元素除以该行的总和，得到每行的概率分布。最终结果存储在 self.probs 中，这个概率分布表示每个样本属于各个类别的概率。
        self.probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return self.probs
    
    def compute_loss(self, X, y):
        # 样本数量
        num_samples = X.shape[0]
        # 前向传播计算概率
        probs = self.forward(X)
        # 计算交叉熵损失 
        corect_logprobs = -np.log(probs[range(num_samples), y]) # probs[range(num_samples), y]是n*1：通过 range(num_samples) 和 y 作为索引，从 probs 矩阵中取出每个样本的真实类别的概率。
        data_loss = np.sum(corect_logprobs)/num_samples
        # 计算正则化损失
        reg_loss = 0.5*self.reg*(np.sum(self.W1**2) + np.sum(self.W2**2))
        return data_loss + reg_loss
    
    def backward(self, X, y, lr):
        num_samples = X.shape[0]
        # 计算第三层误差
        delta3 = self.probs
        delta3[range(num_samples), y] -= 1
        delta3 /= num_samples
        # 计算第二层权重和偏置的梯度
        dW2 = self.a1.T.dot(delta3) + self.reg*self.W2
        db2 = np.sum(delta3, axis=0)
        # 计算第二层误差
        delta2 = delta3.dot(self.W2.T)
        # 根据激活函数类型更新第二层误差
        if self.activation == 'relu':
            delta2[self.z1 <= 0] = 0
        elif self.activation == 'sigmoid':
            delta2 *= (self.a1 * (1-self.a1))
        # 计算第一层权重和偏置的梯度
        dW1 = X.T.dot(delta2) + self.reg*self.W1
        db1 = np.sum(delta2, axis=0)
        
        return dW1, db1, dW2, db2
    
    def update_params(self, dW1, db1, dW2, db2, lr):
        # 更新第一层权重和偏置
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        # 更新第二层权重和偏置
        self.W2 -= lr * dW2 
        self.b2 -= lr * db2

# 训练模块
def train(model, X_train, y_train, X_val, y_val, 
         lr=1e-3, lr_decay=0.95, batch_size=128, epochs=50):
    ''''学习率衰减、随机分批训练、记录训练和验证损失以及准确率，并保存最佳模型'''
    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # 学习率衰减
        if epoch % 10 == 0 and epoch != 0:
            lr *= lr_decay
            
        # 随机分批 
        indices = np.random.permutation(X_train.shape[0]) # 按行随机打乱
        for i in range(0, X_train.shape[0], batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            # 前向传播
            _ = model.forward(X_batch)
            # 反向传播
            grads = model.backward(X_batch, y_batch, lr)
            # 参数更新
            model.update_params(*grads, lr) # 实现了SGD优化器
            # 每次迭代都会调用 backward 方法计算梯度，接着调用 update_params 方法更新参数，构成了SGD优化过程        
        # 记录训练损失 
        train_loss = model.compute_loss(X_train, y_train)
        val_loss = model.compute_loss(X_val, y_val)
        val_acc = np.mean(np.argmax(model.forward(X_val), axis=1) == y_val)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            np.savez('best_model.npz', 
                     W1=model.W1, b1=model.b1,
                     W2=model.W2, b2=model.b2)
            
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    
    return history

# 超参数搜索
def hyperparameter_search(X_train, y_train, X_val, y_val):
    '''用于搜索最佳的超参数组合，通过遍历不同的隐藏层大小、学习率和正则化系数，
       找到验证集准确率最高的参数组合。'''
    best_acc = 0
    best_params = {}
    
    # 待搜索参数组合 
    params_grid = {
        'hidden_size': [512, 1024],  # 更大的网络
        'lr': [1e-3, 3e-4],
        'reg': [1e-4, 5e-4],
    }
    for hs in params_grid['hidden_size']:
        for lr in params_grid['lr']:
            for reg in params_grid['reg']:
                print(f'Testing hs={hs}, lr={lr}, reg={reg}')
                model = ThreeLayerNet(3072, hs, 10, reg=reg)
                history = train(model, X_train, y_train, X_val, y_val,
                                lr=lr, epochs=30)
                max_acc = max(history['val_acc'])
                if max_acc > best_acc:
                    best_acc = max_acc
                    best_params = {'hs': hs, 'lr': lr, 'reg': reg}
    
    print(f'Best val acc: {best_acc:.4f} with params {best_params}')
    return best_params

# 测试模块
def test(model_path, X_test, y_test):
    '''用于测试保存的最佳模型在测试集上的准确率'''
    data = np.load(model_path)
    model = ThreeLayerNet(3072, data['W1'].shape[1], 10)
    model.W1 = data['W1']
    model.b1 = data['b1']
    model.W2 = data['W2']
    model.b2 = data['b2']
    
    test_pred = np.argmax(model.forward(X_test), axis=1)
    acc = np.mean(test_pred == y_test)
    print(f'Test Accuracy: {acc:.4f}')
    return acc

if __name__ == '__main__':
    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10('/opt/tiger/yantu-swift/tasks/suit_match_v2/homework/cifar-10-batches-py')
    # print(f'Train: {X_train.shape}, {y_train.shape}')
    # print(X_train[0], y_train[0])
    # 超参数搜索
    best_params = hyperparameter_search(X_train[:10000], y_train[:10000], 
                                       X_val[:2000], y_val[:2000])
    # out: # Best val acc: 0.3815 with params {'hs': 512, 'lr': 0.001, 'reg': 0.0001}
    # best_params = {'hs': 512, 'lr': 0.001, 'reg': 0.0001}
    # 完整训练
    model = ThreeLayerNet(3072, best_params['hs'], 10, reg=best_params['reg'])
    history = train(model, X_train, y_train, X_val, y_val,
                   lr=best_params['lr'], batch_size=128, epochs=200)
    
    # 结果可视化
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(history['val_acc'])
    plt.title('Validation Accuracy')
    plt.savefig('/opt/tiger/yantu-swift/tasks/suit_match_v2/homework/loss—acc-200.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 测试最佳模型
    test('best_model.npz', X_test, y_test)
    
    # 可视化第一层权重
    W1 = np.load('best_model.npz')['W1']
    print(W1.shape)
    plt.figure(figsize=(10,10))
    for i in range(400):
        plt.subplot(20,20,i+1)
        wimg = (W1[:,i].reshape(32,32,3) - W1[:,i].min()) / (W1[:,i].max() - W1[:,i].min()) 
        plt.imshow(wimg)
        plt.axis('off')
    plt.suptitle('First Layer Weights Visualization')
    plt.savefig('/opt/tiger/yantu-swift/tasks/suit_match_v2/homework/weight-200.png', dpi=300, bbox_inches='tight')
    plt.show()
