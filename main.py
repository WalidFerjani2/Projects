# -*- coding: utf-8 -*-

# author: Thomas Pellegrini
from Bayes import GaussianBayes
import numpy as np
import seaborn as sns
from utils import load_dataset, plot_dataset2d, plot_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, LeaveOneOut, ShuffleSplit


def main():
    gb = GaussianBayes()
    
    csv_chemin = 'data/data2.csv'

    data = load_dataset(csv_chemin)
    
    
    X = []
    y = []
    #y = data[-1]
    #print(y)
    for row in data:
        X.append(row[:-1])
        y.append(row[-1])
    X = np.array(X, dtype=np.float)
    y = np.array(y)
    print(X.shape)
    print(y.shape)
    
    """ KNEIGHBORS METHOD  """ 
    
    X_train, X_test , y_train , y_test = train_test_split(X, y, test_size = 0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    classifier = KNeighborsClassifier(n_neighbors = 10)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    
    
    #cm1 = confusion_matrix(y_test, y_pred)
    #sns.heatmap(data=cm1, fmt='.0f', xticklabels= np.unique(y), yticklabels = np.unique(y), annot=True)
    
    print('accuracy sur base du test' , accuracy_score(y_test, y_pred))
    
    """ BAYES """ 
    
    
    
    train_list , test_list =  gb.split_data(data, weight = .80)
    print("trainl list", len(train_list))
    print("test list", len(test_list))
    
    
    #regroupement de data par label
    group = gb.order_by_label(data)
    
    
    gb.train(train_list, -1)
    
    prediction = gb.predict(test_list)
    score = gb.accuracy(test_list, prediction)
    
    print("score of Naive Bayes classifier")
    print(score)
    
    prediction = np.array(prediction)
    #print(prediction)
    cm2 = confusion_matrix(y_test, prediction)
    sns.heatmap(data=cm2, xticklabels= np.unique(y), yticklabels = np.unique(y), annot=True)
    
    print('accuracy sur base du test' , accuracy_score(y_test, prediction))
    
    #print("grouped into  classes" , len(group.keys()))
    #print(group.keys())
    
    for labels in ['aa', 'ee', 'eh', 'eu', 'ii', 'oe', 'oh', 'oo', 'uu', 'yy']:
        prior = gb.priors(group, labels, data)
        print("P(", labels, ") = ", prior )
    """
    # Strategie de K Fold : Mélanger le dataset puis le découper en 10 sections ( égales )
    # entrainement du modèle et alterner avec les differentes possibilités
    """
    print('CROSS VALIDATION ')
    cv = KFold(8)
    print(cross_val_score(KNeighborsClassifier(), X, y, cv = cv))
    """
    # Strategie de LeaveOneOut 
    """
    #cv = LeaveOneOut()
    #print(cross_val_score(KNeighborsClassifier(), X, y, cv = cv))
    """
    # Strategie de ShouffleSplit : Mélanger le dataset
    # Définir une porition de données pour le test et une portion de données pour l'entrainement
    # Regrouper de nouveau les données 
    # remélanger et recommencer autant de fois 
    """
    #cv = ShuffleSplit(10, test_size=0.20)
    #print(cross_val_score(KNeighborsClassifier(), X, y, cv = cv))
    
    """
    {'eh': {'prior_probability': 0.1025, 'score': 
        [{'mean': 0.5369191097560976, 'standard deviation': 0.0795008516017757}, 
        {'mean': 0.8339530365853659, 'standard deviation': 0.15427781501543542}]}, 
     'uu': {'prior_probability': 0.1, 'score':
         [{'mean': -0.4743846125, 'standard deviation': 0.07154144133129645}, 
         {'mean': 0.4988197124999999, 'standard deviation': 0.11381853174399738}]}, 
     'aa': {'prior_probability': 0.08875, 'score': 
         [{'mean': -0.17871477464788738, 'standard deviation': 0.10626006869888308},
         {'mean': 0.9680267746478873, 'standard deviation': 0.1655626507138782}]}, 
     'eu': {'prior_probability': 0.10625, 'score':
         [{'mean': 0.6030798588235294, 'standard deviation': 0.08405689397113761}, 
         {'mean': 0.4749742000000002, 'standard deviation': 0.12962421531440868}]}, 
     'oe': {'prior_probability': 0.1025, 'score':
         [{'mean': 0.11102251219512198, 'standard deviation': 0.09621453982447588}, 
         {'mean': 0.6044789512195121, 'standard deviation': 0.14124501652075452}]}, 
     'ii': {'prior_probability': 0.09875, 'score': 
         [{'mean': 0.054813784810126574, 'standard deviation': 0.07873849942619555}, 
         {'mean': -0.566977329113924, 'standard deviation': 0.11145018047238053}]},
     'yy': {'prior_probability': 0.10125, 'score': 
         [{'mean': 0.40757444444444446, 'standard deviation': 0.09163553965356455}, 
         {'mean': -0.487307913580247, 'standard deviation': 0.1261994663259514}]},
     'ee': {'prior_probability': 0.1025, 'score': 
         [{'mean': 0.3003121463414634, 'standard deviation': 0.08189377497515928}, 
         {'mean': -0.43441586585365866, 'standard deviation': 0.13408066162932333}]}, 
     'oh': {'prior_probability': 0.1, 'score': 
         [{'mean': 0.022534874999999992, 'standard deviation': 0.10332664038266698}, 
         {'mean': -0.3716429375, 'standard deviation': 0.1475020465320666}]}, 
     'oo': {'prior_probability': 0.0975, 'score': 
         [{'mean': 0.3013596282051284, 'standard deviation': 0.07416537160373239}, 
         {'mean': -0.6890346410256409, 'standard deviation': 0.1365564349815945}]}}"""
          
    
    
    
    
    
    #classifier = KNeighborsClassifier(n_neihbors=10)
    #classifier.fit()
    
    #print(data)
    #print(labels)
    #m, d = data.shape
    #plot_dataset2d(data, labels)
    
    
    
    #g = GaussianBayes(priors=np.zeros(len(np.unique(labels))))

    #g.fit(data, labels)

    #score = g.score(data, labels)

    





if __name__ == '__main__':
    main()
    


