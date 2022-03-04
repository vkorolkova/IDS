import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import time
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from tensorflow import keras
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras import backend as K

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

def map_attack(attack):
    if attack in dos_attacks:
        # dos_attacks map to 1
        attack_type = 1
    elif attack in probe_attacks:
        # probe_attacks mapt to 2
        attack_type = 2
    elif attack in privilege_attacks:
        # privilege escalation attacks map to 3
        attack_type = 3
    elif attack in access_attacks:
        # remote access attacks map to 4
        attack_type = 4
    else:
        # normal maps to 0
        attack_type = 0
        
    return attack_type

dos_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']
probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']
privilege_attacks = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']
access_attacks = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']
attack_labels = ['Normal','DoS','Probe','Privilege','Access']

attack_map = train.label.apply(map_attack)
train['attack_map'] = attack_map

multi_y = train['attack_map']

X = train.drop(['label', 'attack_map'], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, multi_y, test_size=0.2)

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
clf_predictions = clf.predict(x_test) 
train_acc = clf.score(x_train, y_train) 
base_clf_score = accuracy_score(y_test,clf_predictions) 
print("Training accuracy is:", train_acc )
print("Testing accuracy is:", base_clf_score)
print(metrics.classification_report(y_test, clf_predictions, digits=3))

from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators=30)
model3.fit(x_train, y_train)
y_test_pred3 = model3.predict(x_test)
print(metrics.classification_report(y_test, y_test_pred3, digits=3))

from sklearn.ensemble import GradientBoostingClassifier
model6 = GradientBoostingClassifier(random_state=0)
model6.fit(x_train, y_train)
y_test_pred6 = model6.predict(x_test)
print(metrics.classification_report(y_test, y_test_pred6, digits=3))

from sklearn.svm import SVC
model4 = SVC(kernel='rbf', C=2.2)
model4.fit(x_train, y_train)
y_test_pred4 = model4.predict(x_test)
print(metrics.classification_report(y_test, y_test_pred4, digits=3))

from sklearn.linear_model import LogisticRegression
model5 = LogisticRegression(max_iter=1200000, solver='saga')
model5.fit(x_train, y_train)
y_test_pred5 = model5.predict(x_test)
print(metrics.classification_report(y_test, y_test_pred5, digits=3))

from keras.optimizers import Adam
def create_model(learning_rate, num_dense_layers,num_input_nodes,
                 num_dense_nodes, activation, adam_decay):
    #start the model making process and create our first layer
    model = Sequential()
    model.add(Dense(num_input_nodes, activation=activation
                   ))
    #create a loop making a new dense layer for the amount passed to this model.
    #naming the layers helps avoid tensorflow error deep in the stack trace.
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i+1)
        model.add(Dense(num_dense_nodes,
                 activation=activation,
                        name=name
                 ))
    #add our classification layer.
    model.add(Dense(5, activation='softmax'))
    
    #setup our optimizer and compile
    adam = Adam(lr=learning_rate, decay= adam_decay)
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

dummy_y = np_utils.to_categorical(y_train)
dummy_y_test = np_utils.to_categorical(y_test)
gp_model = create_model(0.00033517314578402897, 4, 370, 28, 'relu', 0.009834135347844922)
gp_model.fit(x_train,dummy_y, epochs=50, batch_size=97)
gp_model.evaluate(x_test,dummy_y_test)
nn_prediction = gp_model.predict(x_test)
y_classes = [np.argmax(y, axis=None, out=None) for y in nn_prediction]
print(metrics.classification_report(y_test, y_classes, digits=3))

