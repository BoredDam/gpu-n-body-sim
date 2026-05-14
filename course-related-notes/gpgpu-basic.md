# fondamenti di gpgpu in opencl

## architettura host-device

### host
esegue il programma principale: controlla i dispositivi tramite opportuni comandi

### device
dove avvengono le principali computazioni. fondamentalmente, la gpu.

### compute unit
è il componente specifico del device che esegue la computazione effettiva. ogni processore di una GPU è una compute unit.


## kernel, work-item, work-group

### kernel
è la funzione che può essere eseguita su device. molteplici istanze dello stesso kernel possono essere eseguite (lanciate) in maniera concorrente.

### work-item
un work-item esegue una singola istanza di un kernel.

### work-group
un work-group è un gruppo di work-items che lavorano sulla stessa compute-unit. nel work-group esiste una zona di memoria condivisa tra i vari kernel, nota come "local memory", capace di velocizzare le performance dei work-item che collaborano nello stesso work-group.

ogni work-item ha un id globale e un id locale, dove l'id locale è la posizione relativa al work-group.


## memorie

### global memory
è la memoria del device, accessibile tra tutti i kernel e costante per più lanci di kernel.

### memoria costante
è una memoria read only su device e scritta dall'host.

### local memory
memoria relativa ai work-item dello stesso work-group, ha vita pari al lancio del kernel.

