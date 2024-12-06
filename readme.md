# Optimierung der Newsletter-Öffnungsrate mittels Reinforcement Learning

Dieses Projekt beschreibt die Optimierung von Newsletter-Öffnungsraten durch den Einsatz eines Reinforcement-Learning-Ansatzes. Der Fokus liegt auf der Anpassung von Newsletter-Betreffzeilen an die individuellen Interessen der Kunden.

---

## Agenda

1. Ausgangslage
2. Idee
3. Datengrundlage
4. Verteilung des Trainingsdatensatzes
5. Modellübersicht
6. Leistung
7. Klassifizierung der Testdaten
8. Learnings und Code

---

## Ausgangslage

- Ziel: Kunden öffnen den Newsletter und interagieren mit dem Inhalt (z. B. Klicks auf Links).
- Erfolgsmessung: Öffnungsrate, Klickrate usw.
- Einflussfaktor: Der Betreff hat erheblichen Einfluss auf die Öffnungsrate.

---

## Idee

- Jeder Kunde hat eines der folgenden Interessen:
  - **Anlegen**
  - **Finanzieren**
  - **Vorsorge**
- Ein passender Betreff erhöht die Wahrscheinlichkeit, dass der Kunde den Newsletter öffnet.

---

## Datengrundlage

- Für das Experiment wurden synthetische Daten erstellt:
  - Attribute: Alter, Interessen (Anlegen, Finanzieren, Vorsorge), Groundtruth.
  - Beispiel-Datensatz:
    | Alter | Finanzieren | Anlegen | Vorsorge | Groundtruth  |
    |-------|-------------|---------|----------|--------------|
    | 34    | 0.432       | 0.343   | 0.364    | Finanzieren  |
    | 56    | 0.128       | 0.531   | 0.358    | Anlegen      |
    | 45    | 0.985       | 0.365   | 0.663    | Vorsorge     |

---

## Modellübersicht

- **Ansatz:** DQN (Deep Q-Network) Agent zur Optimierung der Betreffzeilen.
- **State:** Alter und finanzielle Interessen (standardisiert).
- **Actions:** Auswahl eines der drei Finanzthemen.
- **Reward:** 
  - Richtiger Betreff: +3
  - Falscher Betreff: -1

---

## Leistung

![Alt text](presentation/figures/cm_pct.png?raw=true "Clustering Testergebnisse")

- Evaluierung des Modells:
  - Konfusionsmatrix relative Werte.
  - Accuracy: ~75% mit optimierten Parametern.

---

## Klassifizierung der Testdaten

![Alt text](presentation/figures/pca_test_data_action_small.png?raw=true "Clustering Testergebnisse")

- Visualisierung der Testdaten zeigt:
  Die Cluster stimmen grösstenteils mit den Interessen überein.

---

## Learnings

- Die Ziele sollten zu Beginn einfach gehalten werden.
- Die Generierung synthetischer Daten war aufwändig.
- Es gibt viele mögliche Stolpersteine (z. B. Architektur, Hyperparameter, Reward-Funktion).
- Unterstützung durch Tools wie ChatGPT war entscheidend.

---

## Code

Der vollständige Code ist auf GitHub verfügbar: [GitHub Repository](https://github.com/ssch-fpv/lr_newsletter)

---

## Autor

Simon
