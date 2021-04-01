Classification Supervisée

       
M. FERJAN Walid1                               
M. GOMIERO Matthieu



Naive Bayes Classifier est un algorithme très utilisé en Machine Learning.  Il est utilisé pour faire de la classification supervisée. 

La classification ou régression par K-NN ou K-Plus Proches Voisins consiste à attribuer à chaque élément de test une étiquette en fonction de ses voisins directs, suivant une méthode de calcul de distance spécifique. Ici c’est la   distance euclidienne qui est utilisée. 

Naive Bayes et K-NN sont tous les deux, des exemples d’apprentissage supervisé (où les données sont déjà étiquetées) . Dans cet article, on verra la différence entre ces deux algorithmes en termes de précision et coût. 
1.1  Description des données 
Nous partons, pour les tests, d’un signal audio enregistré en studio dans lequel le locuteur prononce 10 voyelles : 
“aa”, “ee”,”eh”,”eu”,”ii”,”oe”,”oh”,”oo”,”uu” et “ yy”. 
Le signal contient une centaine d'occurrences pour chaque voyelle, et la paramétrisation cepstrale a été extraite pour donner 12 coefficients à chaque échantillon. Enfin une Analyse en Composantes Principales a aussi été effectué afin de pouvoir tester sur des dimensions plus petites (2 et 3D) les données.

2  Le classifieur Naive Bayes 
Soient X = ( X1 .. X j) l’ensemble de données et Y la variable à prédire. On considère que Y est catégorielle qui est le cas dans notre exemple, c'est-à-dire qu’on peut discrétiser et découper notre train set en plusieurs intervalles. En apprentissage supervisé, pour un exemple  à classer, la règle bayésienne d’affectation consiste à maximiser la probabilité a posteriori d’appartenance aux classes, c-à-d
y() =yk*arg maxkP[ Y=yk/ X() ]

2.1 Observation des données 
Avant de commencer à classifier les données, il est important de visualiser notre dataset

Nous avons donc 1000 observations de 10 valeurs
2.2 Conception du classifieur 
Dans notre étude sous python, on utilise la méthode 'GaussianBayes' qui suppose que les probabilités des caractéristiques sont gaussiennes :
P(xi |  y ) = 12²exp(-(xi - )²2²)
Les paramètres y et y sont déterminés à partir les méthodes : mean et compute_standard_deviation 
Avec notre jeu de données on peut calculer les probabilités de chaque label: 


Maintenant, qu’on dispose des probabilités a priori, on peut concevoir notre modèle.
Train
Cette méthode est indispensable de faire  l'entraînement sur le train set en calculant la moyenne et l’écart type de chaque groupe de classe pour calculer enfin ‘class likelihoods’.
Likelihoods
C’est le produit de toutes les probabilités normales 
P(x1|’aa’)*P(x2|’aa’)*P(x1|’ee’)*P(x2|’e’e)*...*P(x2|’yy’)
Loi de probabilité à plusieurs variables ( Joint Probability Distribution )
C’est le produit des probabilités calculées et likelihoods, du coup pour chaque classe on calcule les probabilités à priori, on utilise une distribution normale pour chaque feature et on calcule enfin la valeur de la probabilité à plusieurs variables.
Loi de probabilité marginale
C’est la somme de toutes les valeurs de la probabilité à plusieurs variables ( Joint Probability Distribution ) calculées.
La probabilité a posteriori 
c’est la probabilité de chaque classe 
P(class | label ) 
2.3 Apprentissage et prédiction

Après la création du classifieur et l'exécution de prédiction sur la liste de test, on peut noter ces valeurs de précisions pour une valeur de séparation de données est égale à .N en pourcentage.

             Précision en fonction de la taille de séparation pour le dataset2.csv

On en déduit la matrice de confusion pour le classifieur Bayésien 

Matrice de confusion de Naive Bayes
( Dataset2.csv)
On peut penser que les valeurs prédites ne sont pas identiques, mais l’algorithme utilisé estime vraiment le label en fonction des règles de probabilités qu’il a établies.
L’algorithme essaye donc de trouver un classement parfait et identique à la réalité. 
La solution ici est de mesure la qualité de la prédiction par une méthode très rudimentaire, en utilisant la bibliothèque Sckit-Learn qui propose une solution plus aboutie: accuracy_score qui donne un taux de réussite égale à 0.08
2.4 Avantages et limitations 
 En se basant sur la classification des voyelles, on remarque que le classifieur Bayésien est trés rapide car les calculs de probabilités ne sont pas couteux. 

3.1  Le classifieur KNN

Pour déterminer l’appartenance ou non à une classe d’un échantillon de test, on va étudier ses K plus proches voisins en fonction d’une méthode de calcul de distance déterminée à l’avance, ici la distance euclidienne, et trouver la classe la plus présente entre les voisins pour l'attribuer à l’échantillon. 

Soit (X1,Y1),(X2,Y2),...,(Xn,Yn)les couples de données où Y est la classe de labellisation de X, tel que X|Y = r ̴ Prpour r = 1,2(et une loi de distribution de probabilités Pr). Sachant ||.|| la norme euclidienne sur ℝ2et un point x ∈ ℝ2, soit (X(1),Y(1)),(X(n),Y(n))un réarrangement des données d’apprentissage tel que les couples || X(1)-x|| ≤ ...≤ ||X(k)-x|| soient les plus proches voisins appartenant à une même classe.

Nous comparerons la validation croisée à Kblocs (“K Fold Cross Validation”), la validation croisée à un contre tous (“Leave One Out Cross Validation”) et le mélange-sépare (“Shuffle Split”) pour déterminer la meilleure stratégie d’apprentissage sur ces données. 
La validation croisée (“Cross-Validation”) est, en apprentissage automatique, une méthode d’estimation de fiabilité d’un modèle fondé sur une technique d’échantillonnage[1]. Les trois méthodes sont détaillées en annexes de ce document. 

Nous utilisons une distribution des poids uniforme pour cette méthode.

3.2  Etude des résultats du Classifieur KNN


	
Rapport de précision du KNN avec data2.csv

Matrice de confusion KNN avec data2.csv


On observe que plus le nombre de dimensions est élevé, plus la précision générale de l’algorithme augmente.
A partir d’un certain point on arrive même à ne plus avoir d’erreur de classification. 
Seulement augmenter les données n’assure pas une validation croisée réussie, on le voit avec le K Fold Cross Validation qui, même à un nombre élevé de dimensions, peine à atteindre les 0.6 de précision moyenne alors que ses homologues arrive presque à 0.8 de précision moyenne avec seulement 2 dimensions.

	Rapport de précision du KNN avec data3.csv

Matrice de confusion KNN avec data3.csv

Pour les temps de computation on observe que le nombre de dimensions des données est le facteur le plus déterminant, en effet entre les 2-3D et le 12D on peut voir une différence de 0.0005 secondes soit près de 1.5 fois plus long. 
On peut aussi voir que le ratio de vrai-positif (recall) est convenable à partir de 3 dimensions, le gain en recall étant presque négligeable aux dimensions supérieures quand on observe les ratios à 2 dimensions.

	Rapport de précision du KNN avec data12.csv	

Matrice de confusion avec data12.csv



Ainsi à 2 dimensions malgré un score plutôt bon en précision (0.815) le nombre de faux positifs semble trop important pour se permettre de ne pas passer en 3 dimensions, alors que le coût en temps du traitement des 12 dimensions paraît trop important pour le gain qu’il apporte par rapport aux 3 dimensions.



4  Conclusion

Annexe

K Fold Cross Validation : 
Ici on va diviser l’échantillon original en K bloc (qui n’a rien à voir avec le Kdu K-NN) , sélectionner un des blocs comme ensemble de validation et utiliser les autres comme ensemble d’apprentissage. Une fois l’apprentissage réalisé on peut déterminer un performance pour le bloc qui a été choisi et tester un autre bloc. Une fois chaque bloc testé l’on dispose de Kscores de performances, un par bloc, que l’on peut utiliser pour estimer biais et variance de la performance de validation.


Leave One Out Cross Validation : 
Similaire à la K Fold Cross Validation, avec la particularité que chaque bloc sera constitué d’un seul échantillon (d’où K=n), l’apprentissage se réalise sur n-1observations et la validation sur la dernière observation restante. 


Shuffle Split:
Contrairement au K Fold Cross Validation où les paquets sont déterminés à l’avance, le  Shuffle Split va séparer en paquet d’apprentissage et paquet de validation aléatoirement à chaque itération en lui donnant la taille des paquets de tests et de validation que l’on souhaite au départ. On peut ainsi se retrouver à avoir des données sélectionnées pour une itération aussi choisies durant la suivante étant donné qu’on “Split” sur tout le jeu de données à chaque itération.

Bibliographie

[1] https://fr.wikipedia.org/wiki/Validation_croisée

