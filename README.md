# ⚡ Optimisation GPU des calculs de forces entre particules avec Numba CUDA

Ce projet propose une simulation physique de particules interagissant entre elles selon la loi de Coulomb, en comparant les performances d'une version **CPU** classique et d'une version **GPU** optimisée avec **Numba CUDA**.

## 🧠 Objectif
Accélérer des calculs lourds d'interactions entre particules à l’aide de la **parallélisation GPU**, tout en respectant les contraintes suivantes :
- Éviter les multiplications matricielles simples
- Implémenter au moins un **custom kernel** CUDA
- Comparer les performances CPU vs GPU
- Proposer une explication claire et pédagogique

## 🧪 Description de la simulation

Chaque particule possède :
- Une position `(x, y)`
- Une vitesse `(vx, vy)`
- Une charge électrique `q`

Les forces sont calculées entre chaque paire de particules selon la loi de Coulomb simplifiée :
\[
F = \frac{q_i \cdot q_j}{d^2}
\]

Les forces sont ensuite utilisées pour mettre à jour les vitesses et positions.

## ⚙️ Technologies utilisées

- 🐍 Python 3.10+
- 📦 NumPy
- 🚀 Numba (avec `@cuda.jit`)
- 🧮 CUDA (via Numba)
- 🧑‍💻 Jupyter Notebook pour l’explication

## 🧩 Fichiers principaux

- `simulation_cpu_gpu.py` – Code principal de simulation
- `README.md` – Présentation du projet
- `resultats/` – Graphiques ou logs de performances (facultatif)

## 📊 Comparaison des performances

| Version     | Temps d'exécution (3000 particules) |
|-------------|-------------------------------------|
| CPU         | ~7 secondes                         |
| GPU (CUDA)  | ~0.3 secondes                       |

👉 Accélération : environ **20x plus rapide** avec le GPU.

## 🧠 Ce que j’ai appris

- Comment écrire un kernel CUDA personnalisé en Python
- Comment gérer les threads CUDA avec `cuda.grid(1)`
- Optimisation mémoire : `cuda.to_device`, `copy_to_host`
- Importance de la parallélisation dans les simulations physiques

## 🧰 Exécution

```bash
pip install numpy numba
python simulation_cpu_gpu.py
```

## 📌 Auteur

Projet réalisé par **Idriss Elatrech** – Étudiant en 4e année à l’ESIEA
