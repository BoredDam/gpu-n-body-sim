# gpu-n-body-sim
Implementazione parallela su GPU (OpenCL) di algoritmi (esatti e approssimati) per simulazioni a $n$-corpi (a due dimensioni).

## Algoritmo Naive
Noto anche come approccio "brute-force", calcola le interazioni di ciascuna particella con tutte le altre coinvolte nella simulazione, ottenendo una complessità pari a $\mathcal{O}(n^2)$. 

### Integrazione leap-frog
Utilizzeremo in questo ed altri approcci l'integrazione leap-frog, noto metodo numerico per l'integrazione relativa a sistemi dinamici e meccanica classica.
Conserva maggiormente l'energia, a differenza del metodo di Eulero, in cui il sistema guadagna energia artificialmente, rendendo la simulazione meno accurata.

![](https://www.researchgate.net/publication/365371896/figure/fig1/AS:11431281097186284@1668483080156/Mean-numerical-errors-of-the-Euler-and-leapfrog-discretisation-schemes-averaged-during.png)
## Algoritmo Barnes-Hut
Il focus di questo algoritmo è sicuramente la costruzione dei quad-tree.

### Inserimento di un corpo $b$ in un quad-tree
```
1. If node x does not contain a body, put the new body b here.

2.1 If node x is an internal node, update the center-of-mass and total mass of x.
    Recursively insert the body b in the appropriate quadrant.

2.2 If node x is an external node, say containing a body named c, then there are two bodies b and c in the same region.
    Subdivide the region further by creating four children. Then, recursively insert both b and c into the appropriate quadrant(s).
    Since b and c may still end up in the same quadrant, there may be several subdivisions during a single insertion.
    Finally, update the center-of-mass and total mass of x.


from: https://arborjs.org/docs/barnes-hut

```

![](https://jorellano.github.io/img/barnes%20hut.jpg)

Il risultato di questo algoritmo ricorsivo, è un albero in cui ogni nodo contiene al più una particella. Ogni nodo contiene la massa e il centro di massa della regione assegnata. Questo permette, durante il calcolo delle forze che agiscono su una particella, di approssimare regioni più lontane a corpi unici, riducendo la complessità della simulazione a $\mathcal{O}(n\log n)$. Una complessità che ci permette di spingere il numero di particelle della simulazione a valori molto più grandi. Non è un algoritmo imbarazzantemente parallelizzabile, ed è quindi un ottimo caso studio.

