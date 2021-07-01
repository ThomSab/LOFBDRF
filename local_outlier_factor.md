# Local Outlier Factor (LOC)

Ein Algorithmus zur Erkennung von Ausreissern, basierend auf der Dichte der Verteilung.
!Zum groessten Teil aus dem deutschen Wikipedia Artikel zum Local Outlier Factor kopiert.!

Ein Punkt, der „dichter“ ist als seine Nachbarn, befindet sich in einem Cluster.
Ein Punkt mit einer deutlich geringeren Dichte als seine Nachbarn ist hingegen ein Ausreisser.

## k-distance(A)

Die k-distance(A) ist die maximale Distanz des Objektes __A__ zu seinem k nächsten Nachbarn.
<img src="kdistance.png" width="500">\
Diese Menge kann gegebenenfalls mehr als k Objekte enthalten,
wenn es mehrere Objekte mit dem gleichen Abstand gibt.
Wir bezeichnen diese „k-Nachbarschaft“ hier mit __N__<sub>k</sub>__A__

## Reachability Distance rd<sub>k</sub>(A,B)

<strong>rdk(A, B) = max{k−distance(B), d(A, B)}</strong>

Objekte die zu den k-nächsten Nachbarn von __B__ gehören, werden als gleich weit entfernt betrachtet.
Ihre Distanz ist die __k-distance(B)__.
Gehoeren sie nicht zu den k-nächsten Nachbarn is die Reachability Distance ihre wahre Distanz zu __B__: __d(A,B)__.

<img src="rdk.png" width="500">

The rdk of from __A__ to __D__ is larger than that from __A__ to __B__ or __C__.
The rdk  of __B__ and __C__ both is equal to k-distance of __A__.

## Local Reachability Density lrd<sub>k</sub>(A)

Diese Dichte ist der Kehrwert der durchschnittlichen Erreichbarkeitsdistanz des Objektes __A__ von seinen Nachbarn, 
nicht andersherum die durchschnittliche Erreichbarkeitsdistanz (avg. rdk) der Nachbarn von __A__, was definitionsgemäß k-Distanz(__A__) wäre.
<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\Large&space;\bg_white&space;lrd_{k}A:=1/\frac{\sum_{B\in&space;N_{k}(A)}{rd_{k}(A,B)}}{\mid&space;N_{k}(A)\mid}">

## Local Outlier Factor

Vergleicht die lokale Erreichbarkeitsdichte von __A__ jetzt mit denen der Nachbarn.
<img src="https://latex.codecogs.com/gif.latex?\Large&space;\bg_white&space;LOF_{k}(A):=\frac{\sum_{B\in&space;N_{k}(A)}\frac{lrd_{k}(B)}{lrd_{k}(A)}}{\mid&space;N_{k}(A)\mid}&space;=\frac&space;{\sum_{B&space;\in&space;N_{k}(A)}&space;lrd_{k}(B)}{\mid&space;N_{k}(A)\mid}/{lrd}_{k}(A)"
     alt="definition of the local outlier factor of A." >

Der „Local Outlier Factor“ (LOF) ist also die „Durchschnittliche Erreichbarkeitsdichte der Nachbarn“ dividiert durch die Erreichbarkeitsdichte des Objektes (A) selbst.</br>
Ein Wert von etwa __1__ bedeutet, dass das Objekt eine mit seinen Nachbarn vergleichbare Dichte hat (also kein Ausreißer ist).
Ein Wert kleiner als __1__ bedeutet sogar eine dichtere Region (was ein sogenannter „Inlier“ wäre), während signifikant höhere Werte als __1__ einen Ausreißer kennzeichnen.

## Connection to Random Forests: What is the LOF of a Random Tree?

(Ungefaehr aus dem Paper uebersetzt)
Jeder Vorhersage (bezeichnet durch den Vektor C(RF.tree(i), T)) auf dem Trainingsdatensatz (T) die durch einen Random Tree (RF.tree(i)) aus dem Random Forest (RF) gemacht wird,
 wird ein LOF-Wert zugewiesen, der den Grad seiner Ausreißer - heit angibt.

<img src="Tree_LOF.png" width="500">

Da C ein Vektor ist, wird vom Paper impliziert, dass die Distanzberechnung regulaer erfolgt.
Wie verhaelt es sich aber z.B. bei einer multinominalen Klassifizierung?
Wie gross waere der Abstand von Hund zu Katze?



















