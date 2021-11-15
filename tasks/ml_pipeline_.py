import logging
import os
data_path = "/Users/zli/repos/formulation-ml-pipeline/airflow/dags/tasks/data"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
ori_data_path = data_path+"data_file"
train_csv_path = data_path+"train"
test_csv_path = data_path+"test"
train_y_path = str(os.getcwd())+"/train_y"
test_y_path = str(os.getcwd())+"/test_y"
train_x_path = str(os.getcwd())+"/train_x"
test_x_path = str(os.getcwd())+"/test_x"
model_path = str(os.getcwd())+"/model"
output_history_path = str(os.getcwd())+"/output_history"
def download_data():
    import pandas as pd
    # Use pandas excel reader
    df = pd.read_csv(data_path+'ENB2012_data.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(ori_data_path, index=False)

def split_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(ori_data_path)
    train, test = train_test_split(df, test_size=0.2)

    train.to_csv(train_csv_path, index=False)
    test.to_csv(test_csv_path, index=False)

def preprocess_data():
    print(os.getcwd())
    import pandas as pd
    import numpy as np
    import pickle
    
    def format_output(data):
        y1 = data.pop('Y1')
        y1 = np.array(y1)
        y2 = data.pop('Y2')
        y2 = np.array(y2)
        return y1, y2

    def norm(x, train_stats):
        return (x - train_stats['mean']) / train_stats['std']

    train = pd.read_csv(train_csv_path)
    test = pd.read_csv(test_csv_path)

    train_stats = train.describe()

    # Get Y1 and Y2 as the 2 outputs and format them as np arrays
    train_stats.pop('Y1')
    train_stats.pop('Y2')
    train_stats = train_stats.transpose()
    
    train_Y = format_output(train)
    with open(train_y_path, "wb") as file:
      pickle.dump(train_Y, file)
    
    test_Y = format_output(test)
    with open(test_y_path, "wb") as file:
      pickle.dump(test_Y, file)

    # Normalize the training and test data
    norm_train_X = norm(train, train_stats)
    norm_test_X = norm(test, train_stats)


    norm_train_X.to_csv(train_x_path, index=False)
    norm_test_X.to_csv(test_x_path, index=False)

def train_model():
    import pandas as pd
    import tensorflow as tf
    import pickle
    
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Input
    
    norm_train_X = pd.read_csv(train_x_path)

    with open(train_y_path, "rb") as file:
        train_Y = pickle.load(file)
    def model_builder(train_X):

      # Define model layers.
      input_layer = Input(shape=(len(train_X.columns),))
      first_dense = Dense(units='128', activation='relu')(input_layer)
      second_dense = Dense(units='128', activation='relu')(first_dense)

      # Y1 output will be fed directly from the second dense
      y1_output = Dense(units='1', name='y1_output')(second_dense)
      third_dense = Dense(units='64', activation='relu')(second_dense)

      # Y2 output will come via the third dense
      y2_output = Dense(units='1', name='y2_output')(third_dense)

      # Define the model with the input layer and a list of output layers
      model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

      print(model.summary())

      return model

    model = model_builder(norm_train_X)

    # Specify the optimizer, and compile the model with loss functions for both outputs
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss={'y1_output': 'mse', 'y2_output': 'mse'},
                  metrics={'y1_output': tf.keras.metrics.RootMeanSquaredError(),
                          'y2_output': tf.keras.metrics.RootMeanSquaredError()})
    # Train the model for 500 epochs
    history = model.fit(norm_train_X, train_Y, epochs=100, batch_size=10)
    model.save(model_path)

    with open(output_history_path, "wb") as file:
        train_Y = pickle.dump(history.history, file)

def eval_model():
    import pandas as pd
    import tensorflow as tf
    import pickle

    model = tf.keras.models.load_model(model_path)
    
    norm_test_X = pd.read_csv(test_x_path)

    with open(test_y_path, "rb") as file:
        test_Y = pickle.load(file)

    # Test the model and print loss and mse for both outputs
    loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x=norm_test_X, y=test_Y)
    print("Loss = {}, Y1_loss = {}, Y1_mse = {}, Y2_loss = {}, Y2_mse = {}".format(loss, Y1_loss, Y1_rmse, Y2_loss, Y2_rmse))
    
    logging.info("loss", loss)
    logging.info("Y1_loss", Y1_loss)
    logging.info("Y2_loss", Y2_loss)
    logging.info("Y1_rmse", Y1_rmse)
    logging.info("Y2_rmse", Y2_rmse)
