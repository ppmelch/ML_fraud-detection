# Scripts

Scripts operativos para tareas de soporte (ingesta, preparación, utilidades de desarrollo).

- `compare_real_vs_synthetic.py` — re-ejecuta el pipeline sin cambios con solo los datos reales y con el dataset aumentado, y escribe `results/real_vs_synthetic.json` y las figuras `fig13`/`fig14` del paper. Uso: `python -m scripts.compare_real_vs_synthetic`.
