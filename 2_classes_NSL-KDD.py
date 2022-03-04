import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras import backend as K
from keras import Sequential
from keras.regularizers import l2
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","level"]

train = pd.read_csv('KDDTrain+.txt', names = col_names)

train

train.drop(['level'], axis=1, inplace=True)

train = train.dropna('columns')

protocol_type_train = pd.get_dummies(train['protocol_type'], dtype=int)

train.drop(['protocol_type'], axis=1, inplace=True)

train = pd.concat([protocol_type_train, train], axis = 1)

service_type_train = pd.get_dummies(train['service'], dtype=int)

train.drop(['service'], axis=1, inplace=True)

train = pd.concat([service_type_train, train], axis = 1)

flag_type_train = pd.get_dummies(train['flag'], dtype=int)

train.drop(['flag'], axis=1, inplace=True)

train = pd.concat([flag_type_train, train], axis = 1)

Y_train = train[['label']]

Y = np.where(Y_train=='normal',0,1)

X = train.drop(['label'], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

x_train.shape

x_test.shape

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.decomposition import PCA

pca = PCA(n_components=106)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
pred_dt = clf.predict(x_test) 
train_acc = clf.score(x_train, y_train) 
base_clf_score = accuracy_score(y_test,pred_dt) 
print("Training accuracy is:", train_acc )
print("Testing accuracy is:", base_clf_score)
print(metrics.recall_score(y_test, pred_dt))

metrics.precision_score(y_test, pred_dt)

metrics.f1_score(y_test, pred_dt)

from sklearn.ensemble import RandomForestClassifier

model3 = RandomForestClassifier(n_estimators=30)

model3.fit(x_train, y_train)

pred_rf = model3.predict(x_test)

from sklearn.ensemble import GradientBoostingClassifier
model6 = GradientBoostingClassifier(random_state=0)
model6.fit(x_train, y_train)
pred_gb = model6.predict(x_test)

from sklearn.svm import SVC
model4 = SVC(kernel='rbf', C=2.2)
model4.fit(x_train, y_train)
pred_svc = model4.predict(x_test)

from sklearn.linear_model import LogisticRegression

model5 = LogisticRegression(max_iter=1200000, solver='saga')
model5.fit(x_train, y_train)

pred_lr = model5.predict(x_test)

dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')
dim_num_input_nodes = Integer(low=1, high=512, name='num_input_nodes')
dim_num_dense_nodes = Integer(low=1, high=28, name='num_dense_nodes')
dim_activation = Categorical(categories=['relu', 'sigmoid'], name='activation')
dim_batch_size = Integer(low=1, high=128, name='batch_size')
dim_adam_decay = Real(low=1e-6,high=1e-2,name="adam_decay")

dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_input_nodes,
              dim_num_dense_nodes,
              dim_activation,
              dim_batch_size,
              dim_adam_decay
             ]
default_parameters = [1e-3, 1,512, 13, 'relu',64, 1e-3]

from keras.optimizers import Adam
def create_model(learning_rate, num_dense_layers,num_input_nodes,
                 num_dense_nodes, activation, adam_decay):
    #start the model making process and create our first layer
    model = Sequential()
    model.add(Dense(num_input_nodes, activation=activation))
    #create a loop making a new dense layer for the amount passed to this model.
    #naming the layers helps avoid tensorflow error deep in the stack trace.
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i+1)
        model.add(Dense(num_dense_nodes,
                 activation=activation,
                        name=name))
    #add our classification layer.
    model.add(Dense(1, activation='sigmoid'))
    
    #setup our optimizer and compile
    adam = Adam(lr=learning_rate, decay= adam_decay)
    model.compile(optimizer=adam, loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_input_nodes, 
            num_dense_nodes,activation, batch_size,adam_decay):

    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_input_nodes=num_input_nodes,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation,
                         adam_decay=adam_decay
                        )
    #named blackbox becuase it represents the structure
    blackbox = model.fit(x=X_train,
                        y=Y_train,
                        epochs=50,
                        batch_size=batch_size,
                        validation_split=0.15,
                        )
    #return the validation accuracy for the last epoch.
    accuracy = blackbox.history['val_accuracy'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()
    # Delete the Keras model with these hyper-parameters from memory.
    del model
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    tensorflow.reset_default_graph()
    # the optimizer aims for the lowest score, so we return our negative accuracy
    return -accuracy

import tensorflow
gp_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            n_calls=12,
                            noise= 0.01,
                            n_jobs=-1,
                            kappa = 5,
                            x0=default_parameters)

K.clear_session()
tensorflow.reset_default_graph()
gp_result.fun
gp_result.x_iters
gp_model = create_model(0.00896405688603655, 2, 382, 27, 'relu', 0.003356381112367726)
gp_model.fit(x_train,y_train, epochs=50, batch_size=65)
gp_model.evaluate(x_test,y_test)

pred_gp = gp_model.predict(x_test)
pred_gp[pred_gp <= 0.5] = 0
pred_gp[pred_gp > 0.5] = 1

ns_probs = [0 for _ in range(len(y_test))]

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

ns_auc = roc_auc_score(y_test, ns_probs)
auc_dt = roc_auc_score(y_test, pred_dt)
auc_rf = roc_auc_score(y_test, pred_rf)
auc_gb = roc_auc_score(y_test, pred_gb)
auc_svc = roc_auc_score(y_test, pred_svc)
auc_lr = roc_auc_score(y_test, pred_lr)
auc_gp = roc_auc_score(y_test, pred_gp)

print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (auc_dt))
print('Random Forest: ROC AUC=%.3f' % (auc_rf))
print('Boosting: ROC AUC=%.3f' % (auc_gb))
print('SVC: ROC AUC=%.3f' % (auc_svc))
print('Logistic: ROC AUC=%.3f' % (auc_lr))
print('NN: ROC AUC=%.3f' % (auc_gp))

ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(y_test, pred_dt)
rf_fpr, rf_tpr, _ = roc_curve(y_test, pred_rf)
gb_fpr, gb_tpr, _ = roc_curve(y_test, pred_gb)
svc_fpr, svc_tpr, _ = roc_curve(y_test, pred_svc)
lr_fpr, lr_tpr, _ = roc_curve(y_test, pred_lr)
gp_fpr, gp_tpr, _ = roc_curve(y_test, pred_gp)

pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')
pyplot.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest')
pyplot.plot(gb_fpr, gb_tpr, marker='.', label='Boosting')
pyplot.plot(svc_fpr, svc_tpr, marker='.', label='SVC')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
pyplot.plot(gp_fpr, gp_tpr, marker='.', label='Neural Network')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

