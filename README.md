# Przetwarzanie obrazów pochodzących z zbioru CIFAR10

Badanie architektur sieci neuronowych do klasyfikacji obrazów na zbiorze CIFAR-10 przy użyciu PyTorch.

## Opis projektu

Projekt porównuje kilka podejść — od prostej sieci w pełni połączonej po głębsze sieci CNN
z normalizacją, dropoutem i regularyzacją L2 — na zbiorze CIFAR-10
(60 000 kolorowych obrazów 32×32 px, 10 klas).

**Najlepszy wynik: ~75%+ dokładności na zbiorze testowym** (3-warstwowa CNN, 20 epok).

## Wyniki

| Model | Architektura | Regularyzacja | Dokładność testowa |
|-------|-------------|---------------|--------------------|
| Baseline FC | 3× Linear | — | ~40% |
| ConvNet v1 | 2× Conv + FC | BatchNorm | ~65% |
| ConvNet v2 | 3× Conv + klasyfikator | Dropout 0.5, Weight Decay | ~75%+ |

## Architektura najlepszego modelu
```
Wejście (3×32×32)
→ Conv(3→32) + BN + ReLU + MaxPool
→ Conv(32→64) + BN + ReLU + MaxPool
→ Conv(64→128) + BN + ReLU + MaxPool
→ Linear(2048→256) + ReLU + Dropout(0.5)
→ Linear(256→10)
```

## Instalacja

```bash
pip install torch torchvision matplotlib
```

Zbiór CIFAR-10 pobierany jest automatycznie przez torchvision przy pierwszym uruchomieniu.

## Uruchomienie

Aby uruchomić należy otworzyć plik `przetwarzanie_obrazow_cifar10.ipynb` w Jupyter Notebook i uruchomić wszystkie komórki.

## Wnioski

- Sieci w pełni połączone osiągają ok. 40% — niewystarczające dla danych przestrzennych
- Normalizacja (BatchNorm) istotnie stabilizuje proces uczenia
- 3 bloki konwolucyjne ze wzrastającą liczbą filtrów (32→64→128) przewyższają mniejsze sieci
- Dropout (p=0.5) w klasyfikatorze skutecznie ogranicza przeuczenie

## Technologie

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
