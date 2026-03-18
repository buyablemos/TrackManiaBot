# Trackmania RL Agent 🏎️🤖

Projekt realizowany w ramach kursu **Sztuczna inteligencja w grach komputerowych**.

## Autorzy
* **Jakub Klimek** (nr indeksu: 263845)
* **Dawid Pyta** (nr indeksu: 263514)
* **Marta Skowron** (nr indeksu: 268400)

## Opis Projektu
Celem projektu jest stworzenie autonomicznego bota do gry **Trackmania**, który nauczy się optymalnej linii przejazdu przy wykorzystaniu technik **uczenia ze wzmocnieniem (Reinforcement Learning)**. 

### Dlaczego Trackmania?
* **Przewidywalna fizyka:** Gra posiada deterministyczny silnik, co ułatwia proces uczenia modelu.
* **Narzędzia:** Wykorzystujemy bibliotekę `TMInterface` do komunikacji między skryptami Python a silnikiem gry.
* **Benchmark:** Finalnym testem będzie porównanie wyników bota z czasami przejazdów uzyskanymi przez autorów projektu.

---

## Harmonogram i Kamienie Milowe

| Faza | Termin końcowy | Opis i cele |
| :--- | :--- | :--- |
| **I** | **14.04.2026** | **KM1: Setup Środowiska.** Konfiguracja `TMInterface`, połączenie Python-gra, definicja prostej przestrzeni obserwacji (pozycja, prędkość). |
| **II** | **05.05.2026** | **KM2: Trening i Reward Function.** Implementacja algorytmu RL, dopracowanie funkcji nagrody, uzyskanie pierwszych pełnych przejazdów trasy. |
| **III** | **02.06.2026** | **KM3: Optymalizacja i Testy.** Fine-tuning modelu, próba pobicia rekordów autorów, przygotowanie końcowej dokumentacji wyników. |


## Technologie
* **Język:** Python
* **RL Framework:** Stable Baselines3 / PyTorch
* **Komunikacja:** TMInterface
* **Gra:** Trackmania
