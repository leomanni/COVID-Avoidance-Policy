# NOTE GENERALI

Il problema è episodico, a singolo agente. L'obiettivo è spostare *n* persone entro una mappa gridworld da una posizione iniziale casuale ad una finale prefissata.
Il reward finale è complessivo delle mosse effettuate (i.e. del tempo impiegato), dell'eventuale scorretto posizionamento delle persone (ossia, ci sono delle mosse effettivamente illegali) e dei contagi avvenuti.
Durante i movimenti, la probabilità d'infezione dipende dalla distanza, cioè se due persone o più persone si trovano entro un certo raggio da un infetto hanno ad ogni istante di tempo una certa probabilità di contrarre la malattia. Il numero di persone inizialmente infette in ogni episodio deve essere casuale. **Ad oggi 3/7/2021 questa feature verrà aggiunta solo successivamente, non appena la correttezza del modello e della sua implementazione saranno state verificate.**
Per l'implementazione è consigliabile provare ad utilizzare le Reinforcement Learning Toolbox e Deep Learning Toolbox.

## DA FARE

- [ ] Definire azioni "impossibili" da mappare in uno stato terminale unico "sconfitta". (Leonardo)

- [ ] Definire la mappa, con ostacoli di confine ed ostacoli interni. Mappe per 4 (circa 6x6), e poi una più grande. (Edoardo)
- [ ] Studiare implementazione environment custom in RL Toolbox (Create Custom MATLAB Environments from Template) (Roberto).
- [ ] Studiare implementazione agente custom in RL Toolbox (Create Custom Reinforcement Learning Agents) (Emanuele, Filippo).
- [ ] Definire codifica stati.
- [ ] Definire codifica azioni.

## Environment e sua dinamica

Si tratta di un gridworld, modellato come una tabella quadrata in cui le caselle "libere da ostacoli" definiscono un ambiente in cui le persone sono libere di spostarsi, potenzialmente attorno a degli oggetti immobili ("ostacoli interni"). La mappa effettiva è dunque circondata da uno strato di ostacoli che la rende quadrata. I landmark iniziali sono decisi randomicamente ad ogni episodio mentre quelli finali sono predeterminati. Vi sono due stati terminali: "vittoria" ossia "tutti nei loro target", e "sconfitta" che si verifica come specificato sotto.
Detto ciò, cosa succede se:

- Ci si muove verso un ostacolo: si resta dove si è. **OCCHIO: Questa cosa deve essere modellata in modo da risultare peggiore che essere stati contagiati.**
- Si verifica uno *stallo* i.e. l'agente mantiene tutti fermi: in base all'epsilon della politica di esplorazione, si definisce una soglia di volte che l'azione "tutti fermi" può essere scelta consecutivamente, oltre la quale l'episodio viene terminato nello stato "sconfitta".
- Si esegue una mossa illegale: si termina nello stato "sconfitta".

### Codifica mappa

Matrice quadrata *M* con elementi scelti secondo:
- Ostacolo: -1
- Libero: 0
- Occupata: *i*, ID del giocatore *i*-esimo per 1 <= *i* <= *n*.

### Codifica stati

Posizione delle singole persone nella mappa, come ad esempio numero della casella occupata in column-major order (purtroppo Matlab è così...).

### Codifica azioni

_n_ numeri da 1 a 5, ciascuno indicante un'azione tra STOP-NSWE.

Le azioni impossibili sono:

## Reward signal

- Ad ogni istante di tempo non terminale: -1 * n_non_in_target.
- Nello stato terminale "sconfitta": reward commisuratamente molto negativo.
- Nello stato terminale "vittoria": -100 * (n_infetti - n_infetti_init)

## Struttura agente

E' consigliabile usare gli algoritmi SARSA/DQN basati su NN implementati in Matlab.

## Rete neurale per approssimazione funzione *q(s, a)*

Semplice rete a sigmoide con pesi come nell'esempio del TD-Gammon.

### Hidden layers ed hidden units

Dovrebbe bastare un unico hidden layer. Testare numero hidden units, iniziare da _2*(n_stati + n_azioni)_.

### Input units

_2n_ input units: le prime _n_ codificano la posizione, in column-major order nella matrice _M_ della mappa, di ciascuna persona, le seconde _n_ codificano con numeri da 1 a 5 le azioni da impartire a ciascuna persona.