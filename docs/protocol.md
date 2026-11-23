
# Protocole USB LiteVNA 64 (binaire NanoVNA V2 / SAA2)

Source de vérité : annexe « USB data interface » du manuel LiteVNA (firmware
≥ 1.3) et l'implémentation de référence `NanoVNA-Saver/Hardware/NanoVNA_V2.py`
(GPL — à lire comme documentation, ne pas copier le code).

## Transport
- USB CDC-ACM, `/dev/ttyACM0` (udev → `/dev/litevna`). Le baudrate est ignoré
  (USB natif) ; `pyserial` avec `timeout` ≥ 1 s.
- Tout est little-endian.
- Avant de parler à l'appareil : envoyer 8 octets `0x00` pour resynchroniser
  l'automate de commandes, puis vider le buffer de réception.

## Commandes hôte → appareil
| Opcode | Nom       | Format                          | Réponse                    |
|--------|-----------|---------------------------------|----------------------------|
| 0x00   | NOP       | —                               | aucune                     |
| 0x0D   | INDICATE  | —                               | 1 octet `0x32` ('2')       |
| 0x10   | READ      | addr(u8)                        | 1 octet                    |
| 0x11   | READ2     | addr(u8)                        | 2 octets                   |
| 0x12   | READ4     | addr(u8)                        | 4 octets                   |
| 0x18   | READFIFO  | addr(u8), n(u8)                 | n × 32 octets (addr=0x30)  |
| 0x20   | WRITE     | addr(u8), val(u8)               | aucune                     |
| 0x21   | WRITE2    | addr(u8), val(u16)              | aucune                     |
| 0x22   | WRITE4    | addr(u8), val(u32)              | aucune                     |
| 0x23   | WRITE8    | addr(u8), val(u64)              | aucune                     |
| 0x28   | WRITEFIFO | addr(u8), n(u8), n octets       | aucune                     |

READFIFO : `n` ≤ 255 entrées par commande ; pour un sweep de N points, boucler.

## Registres
| Adresse | Taille | Nom                | Rôle                                             |
|---------|--------|--------------------|--------------------------------------------------|
| 0x00    | u64    | sweepStartHz       | Fréquence de départ (Hz)                         |
| 0x10    | u64    | sweepStepHz        | Pas de fréquence (Hz)                            |
| 0x20    | u16    | sweepPoints        | Nombre de points (1–65535)                       |
| 0x22    | u16    | valuesPerFrequency | Mesures par point (averaging matériel)           |
| 0x26    | u8     | sweepMode (LiteVNA)| voir manuel                                       |
| 0x30    | FIFO   | valuesFIFO         | Flux de mesures, entrées de 32 octets            |
| 0x40    | u8     | averageSetting     | LiteVNA : moyenne (1,2,4… voir manuel)           |
| 0x41    | u8     | si5351Power        | Puissance source < ~140 MHz                      |
| 0x42    | u8     | adf4350Power       | Puissance source > ~140 MHz                      |
| 0xF0    | u8     | deviceVariant      | 0x02 = SAA2 / LiteVNA                            |
| 0xF1    | u8     | protocolVersion    | 0x01                                             |
| 0xF2    | u8     | hardwareRevision   |                                                  |
| 0xF3    | u8     | firmwareMajor      |                                                  |
| 0xF4    | u8     | firmwareMinor      |                                                  |

Écrire un registre du sweep (0x00/0x10/0x20/0x22) **vide le FIFO** et relance
un sweep. Écrire `0x30` (n'importe quelle valeur, WRITE 1 octet) vide le FIFO
sans changer la configuration.

Fréquence du point k : `f_k = sweepStartHz + k * sweepStepHz`.
Pour un sweep start/stop/N : `step = (stop - start) / (N - 1)` (entier Hz).

## Format d'une entrée FIFO (32 octets)
| Offset | Type  | Champ      |
|--------|-------|------------|
| 0      | i32   | fwd0Re     |
| 4      | i32   | fwd0Im     |
| 8      | i32   | rev0Re     |
| 12     | i32   | rev0Im     |
| 16     | i32   | rev1Re     |
| 20     | i32   | rev1Im     |
| 24     | u16   | freqIndex  |
| 26     | 6 × u8| réservé    |

`struct` Python : `"<iiiiiiH6x"`.

```
fwd0 = fwd0Re + 1j*fwd0Im      # référence (onde incidente port 1)
rev0 = rev0Re + 1j*rev0Im      # réfléchi port 1
rev1 = rev1Re + 1j*rev1Im      # transmis port 2
S11_raw = rev0 / fwd0
S21_raw = rev1 / fwd0
```

Si `valuesPerFrequency > 1`, l'appareil envoie plusieurs entrées avec le même
`freqIndex` : moyenner S11_raw / S21_raw (ou moyenner les complexes bruts
avant division) côté hôte. Les entrées peuvent arriver dans le désordre après
une resynchro : toujours ranger par `freqIndex`.

## Séquence d'un sweep complet
1. Resync (8 × `0x00`), flush RX, INDICATE → attendre `0x32`.
2. Lire 0xF0/0xF1/0xF3/0xF4, refuser si `deviceVariant != 2`.
3. WRITE8 0x00 start, WRITE8 0x10 step, WRITE2 0x20 N, WRITE2 0x22 avg.
4. Relire 0x20 et 0x22 pour vérifier.
5. Boucle READFIFO 0x30 par paquets ≤ 255 jusqu'à avoir N × avg entrées.
   Timeout global ≈ `N * avg / 500 + 2` secondes.
6. Assembler, moyenner, retourner `np.ndarray` complexe shape (N, 2) + fréquences.

## Calibration
L'appareil renvoie des données brutes. Calibration SOLT (1 port : Open/Short/
Load ; 2 ports : + Thru) à faire côté hôte, par exemple avec
`skrf.calibration.OnePort` / `skrf.calibration.SOLT`, kit idéal en première
approche (les standards fournis ne sont pas caractérisés).
