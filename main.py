import numpy as np 
from sklearn import linear_model           
from sklearn.metrics import accuracy_score 
from skimage import io
import matplotlib.pyplot as plt
from skimage.transform import resize
np.random.seed(2)

# Test view
Img = io.imread('Face_data/Stirling/f2v2e1.gif')
print(Img.shape)
imgPlot = plt.imshow(Img, cmap='gray')
plt.show()


person_id = np.arange(1,19)
ver_id = np.arange(1,4)
exp_id = np.arange(1,4)

# Find the consistent size
f_img_shape = []
for p in person_id:
    for v in ver_id:
        for e in exp_id:
            file = 'f' + str(p) + 'v' + str(v) + 'e' + str(e)
            file_read = 'Face_data/Stirling/' + file +'.gif'
            try: 
                img = io.imread(file_read)
                f_img_shape.append(img.shape)
            except FileNotFoundError:
                continue
f_img_shape = np.array(f_img_shape)

m_img_shape = []
for p in person_id:
    for v in ver_id:
        for e in exp_id:
            file = 'm' + str(p) + 'v' + str(v) + 'e' + str(e)
            file_read = 'Face_data/Stirling/' + file +'.gif'
            try: 
                img = io.imread(file_read)
                m_img_shape.append(img.shape)
            except FileNotFoundError:
                continue
m_img_shape = np.array(m_img_shape)

D = 346*266
d = 800


ProjectionMatrix = np.random.randn(D, d) 

# Convert pictures to data 
f_X = np.zeros((162,D))
f_y = np.zeros(162,)
m_X = np.zeros((150,D))
m_y = np.zeros(150,)

i = 0
for p in person_id:
    for v in ver_id:
        for e in exp_id:
            file = 'f' + str(p) + 'v' + str(v) + 'e' + str(e)
            file_read = 'Face_data/Stirling/' + file +'.gif'
            try: 
                img = io.imread(file_read)
                img = resize(img, (346, 266))
                img = img.reshape(346*266,)
                f_X[i,:] = img 
                f_y[i] = 1
                i+=1
            except FileNotFoundError:
                continue
            
i = 0
for p in person_id:
    for v in ver_id:
        for e in exp_id:
            file = 'm' + str(p) + 'v' + str(v) + 'e' + str(e)
            file_read = 'Face_data/Stirling/' + file +'.gif'
            try: 
                img = io.imread(file_read)
                img = resize(img, (346, 266))
                img = img.reshape(346*266,)
                m_X[i,:] = img 
                m_y[i] = 0
                i+=1
            except FileNotFoundError:
                continue
            

X_full = np.concatenate((f_X, m_X), axis=0)
y_full = np.concatenate((f_y, m_y))             # label y

# Feature Engineering
X_train = np.dot(X_full, ProjectionMatrix)

x_mean = X_train.mean(axis = 0)
x_var  = X_train.var(axis = 0)

def feature_extraction(X):
    return (X - x_mean)/x_var 

X_data = feature_extraction(X_train)             # data X

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_full, test_size=20)

# Applied into model
logreg = linear_model.LogisticRegression(C=1e5,max_iter=500)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print ("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))




##############
person_id = np.arange(1,17)
ver_id = np.arange(1,4)
exp_id = np.arange(1,4)

D = 346*266
d = 800

np.random.seed(2)
ProjectionMatrix = np.random.randn(D, d) 

# Convert pictures to data 
f_X = np.zeros((144,D))
f_y = np.zeros(144,)
m_X = np.zeros((132,D))
m_y = np.zeros(132,)

i = 0
for p in person_id:
    for v in ver_id:
        for e in exp_id:
            file = 'f' + str(p) + 'v' + str(v) + 'e' + str(e)
            file_read = 'Face_data/Stirling/' + file +'.gif'
            try: 
                img = io.imread(file_read)
                img = resize(img, (346, 266))
                img = img.reshape(346*266,)
                f_X[i,:] = img 
                f_y[i] = 1
                i+=1
            except FileNotFoundError:
                continue
            
i = 0
for p in person_id:
    for v in ver_id:
        for e in exp_id:
            file = 'm' + str(p) + 'v' + str(v) + 'e' + str(e)
            file_read = 'Face_data/Stirling/' + file +'.gif'
            try: 
                img = io.imread(file_read)
                img = resize(img, (346, 266))
                img = img.reshape(346*266,)
                m_X[i,:] = img 
                m_y[i] = 0
                i+=1
            except FileNotFoundError:
                continue 


X_train = np.concatenate((f_X, m_X), axis=0)
y_train = np.concatenate((f_y, m_y))             # label y

# Feature Engineering
X_train = np.dot(X_train, ProjectionMatrix)

x_mean = X_train.mean(axis = 0)
x_var  = X_train.var(axis = 0)

def feature_extraction(X):
    return (X - x_mean)/x_var 

X_train = feature_extraction(X_train)  


person_id = np.arange(16,19)
ver_id = np.arange(1,4)
exp_id = np.arange(1,4)

f_X_test = np.zeros((27,D))
f_y_test = np.zeros(27,)
m_X_test = np.zeros((27,D))
m_y_test = np.zeros(27,)

i = 0
for p in person_id:
    for v in ver_id:
        for e in exp_id:
            file = 'f' + str(p) + 'v' + str(v) + 'e' + str(e)
            file_read = 'Face_data/Stirling/' + file +'.gif'
            try: 
                img = io.imread(file_read)
                img = resize(img, (346, 266))
                img = img.reshape(346*266,)
                f_X_test[i,:] = img 
                f_y_test[i] = 1
                i+=1
            except FileNotFoundError:
                continue
            
i = 0
for p in person_id:
    for v in ver_id:
        for e in exp_id:
            file = 'm' + str(p) + 'v' + str(v) + 'e' + str(e)
            file_read = 'Face_data/Stirling/' + file +'.gif'
            try: 
                img = io.imread(file_read)
                img = resize(img, (346, 266))
                img = img.reshape(346*266,)
                m_X_test[i,:] = img 
                m_y_test[i] = 0
                i+=1
            except FileNotFoundError:
                continue 
            
X_test = np.concatenate((f_X_test, m_X_test), axis=0)
y_test = np.concatenate((f_y_test, m_y_test))             # label y

# Feature Engineering
X_test = np.dot(X_test, ProjectionMatrix)
X_test = feature_extraction(X_test)  

# Result
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print ("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))


def feature_extraction_fn(fn):
    """
    extract feature from filename
    """
    # vectorize
    img = io.imread(fn)
    img = resize(img, (346, 266))
    img = img.reshape(1,-1)
    # project
    im1 = np.dot(img, ProjectionMatrix)
    # standardization 
    return feature_extraction(im1)

path = 'Face_Data/Stirling/'
fn1 = path + 'f17v3e3.gif'

x1 = feature_extraction_fn(fn1)
p1 = logreg.predict_proba(x1)
print(p1)

def display_result(fn):
    x1 = feature_extraction_fn(fn)
    p1 = logreg.predict_proba(x1)
    img = io.imread(fn)
    
    
    fig = plt.figure()
#     gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
#     plt.subplot(1, 2, 1)
    plt.figure(facecolor="white")
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(img,cmap='gray')
#     plt.axis('off')
#     plt.show()
    plt.subplot(122)
    plt.barh([0, 1], p1[0], align='center', alpha=0.9)
    plt.yticks([0, 1], ('M', 'W'))
    plt.xlim([0,1])
    plt.show()

display_result(fn1)


i = 0
for p in person_id:
    for v in ver_id:
        for e in exp_id:
            file = 'f' + str(p) + 'v' + str(v) + 'e' + str(e)
            file_read = 'Face_data/Stirling/' + file +'.gif'
            display_result(file_read)
            



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = linear_model.LogisticRegression(C=1e5, max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)
print ("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))


logreg = linear_model.LogisticRegression(C=1e5, max_iter=10000, solver='saga')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print ("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
