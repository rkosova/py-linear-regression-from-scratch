import numpy as np

class LinearRegression:
                
    def train(self, X_train, y_train, batch_size, lr):
        self.X = X_train
        self.y = y_train
        self.batch_size = batch_size

        self.batched_X = np.array_split(self.X, np.arange(self.batch_size, len(self.X), self.batch_size))
        self.batched_y = np.array_split(self.y, np.arange(self.batch_size, len(self.y), self.batch_size))

        self.w = np.random.randn() * 0.0001
        self.b = np.random.randn() * 0.001

        
        for i in range(len(self.batched_X)):
            pred = self.predict(self.batched_X[i])
            loss = np.mean(np.power(pred - self.batched_y[i], 2)/2)
            loss_w_grad = 2 * (pred - self.batched_y[i]) * self.batched_X[i]
            loss_b_grad = 2 * (pred - self.batched_y[i])


            loss_w_mean_grad = np.mean(loss_w_grad)
            loss_b_mean_grad = np.mean(loss_b_grad)

            self.w -= (lr * loss_w_mean_grad)
            self.b -= (lr * loss_b_mean_grad)

            print(f"Avg. Loss in Batch: {loss:.5f} [{i+1}/{len(self.batched_X)} { '=' * round( 50 * ( i/len(self.batched_X[i]) ) ) }]", end='\r')

        return loss
    


    def predict(self, X):
        return self.w * X + self.b

