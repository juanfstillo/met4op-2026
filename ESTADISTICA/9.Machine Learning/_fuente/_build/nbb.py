"""Helper mínimo para construir los notebooks de la unidad 9 y ejecutarlos.

Uso:
    from nbb import md, code, write_nb
    write_nb([md("# Titulo"), code("print(1)")], "ruta/al/notebook.ipynb")
"""
import hashlib
import itertools
import json
import os
import subprocess
import sys

PYR = r"C:\Users\mmill\anaconda3\envs\PYR\python.exe"
REPO = r"c:\Users\mmill\OneDrive - moiguer.com\MIOS_PERSONAL\MET4OP - Entorno\met4op-2026"

_contador = itertools.count()


def _nuevo_id(source):
    """id estable y unico por celda (nbformat >= 4.5 lo exige)."""
    h = hashlib.sha1(f"{next(_contador)}::{source}".encode("utf-8")).hexdigest()
    return h[:12]


def md(source):
    return {"cell_type": "markdown", "id": _nuevo_id(source),
            "metadata": {}, "source": _split(source)}


def code(source):
    return {
        "cell_type": "code",
        "id": _nuevo_id(source),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split(source),
    }


def _split(source):
    source = source.strip("\n")
    lines = source.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


# Kernel neutro: el notebook no debe atarse al entorno de esta maquina.
# Cada estudiante lo abre con el kernel de su propio entorno.
KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}


def requisitos(extras=()):
    """Celda de requisitos: nunca instala, solo documenta y verifica."""
    base = ["numpy", "pandas", "matplotlib", "seaborn", "scikit-learn", "statsmodels"]
    paquetes = base + list(extras)
    pip = " ".join(paquetes)
    # nombre de import != nombre de paquete en algunos casos
    imports = {"scikit-learn": "sklearn", "opencv-python": "cv2"}
    lista = ", ".join(repr(imports.get(p, p)) for p in paquetes)
    return code(
        "# Requisitos de este notebook.\n"
        "# Si te falta alguno, instalalo EN TU PROPIO ENTORNO desde una terminal\n"
        "# (o descomentando la linea de abajo) y reinicia el kernel:\n"
        f"#\n"
        f"#     pip install {pip}\n"
        f"#\n"
        f"# %pip install {pip}\n"
        "\n"
        "import importlib\n"
        "\n"
        f"_requeridos = [{lista}]\n"
        "_faltan = []\n"
        "for _m in _requeridos:\n"
        "    try:\n"
        "        _mod = importlib.import_module(_m)\n"
        "        print(f\"  {_m:16s} {getattr(_mod, '__version__', 'sin version')}\")\n"
        "    except ImportError:\n"
        "        _faltan.append(_m)\n"
        "        print(f\"  {_m:16s} FALTA\")\n"
        "\n"
        "if _faltan:\n"
        "    print(f\"\\nFaltan {len(_faltan)} paquetes: {', '.join(_faltan)}.\")\n"
        "    print(\"Instalalos en tu entorno antes de seguir.\")\n"
        "else:\n"
        "    print(\"\\nTodo listo.\")"
    )


def write_nb(cells, path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": KERNELSPEC,
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"escrito {path}")
    print(f"  celdas: {len(cells)} ({n_code} codigo, {len(cells) - n_code} markdown)")
    return path


def execute(path, timeout=1200):
    """Ejecuta el notebook in-place con el kernel pyr. Devuelve True si no hubo errores."""
    cmd = [
        PYR, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        "--ExecutePreprocessor.kernel_name=pyr",
        f"--ExecutePreprocessor.timeout={timeout}",
        path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    err = (r.stderr or "") + (r.stdout or "")
    if r.returncode != 0:
        print("FALLO la ejecucion:")
        print(err[-6000:])
        return False
    normalizar_kernel(path)
    print(f"ejecutado OK: {os.path.basename(path)}")
    return True


def normalizar_kernel(path):
    """Devuelve el kernelspec a algo neutro: ejecutar con el env local no debe
    quedar grabado en el notebook que reciben los estudiantes."""
    nb = json.load(open(path, encoding="utf-8"))
    nb["metadata"]["kernelspec"] = dict(KERNELSPEC)
    li = nb["metadata"].get("language_info", {})
    for clave in ("version", "codemirror_mode", "pygments_lexer",
                  "nbconvert_exporter", "file_extension", "mimetype"):
        li.pop(clave, None)
    nb["metadata"]["language_info"] = {"name": "python"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)


def report(path):
    """Resumen post-ejecucion: celdas sin output y errores."""
    nb = json.load(open(path, encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    sin_out, errores = [], []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        outs = c.get("outputs", [])
        if not outs:
            sin_out.append(i)
        for o in outs:
            if o.get("output_type") == "error":
                errores.append((i, o.get("ename"), " ".join(o.get("evalue", "").split())[:200]))
    print(f"{os.path.basename(path)}: {len(nb['cells'])} celdas, {len(code_cells)} de codigo")
    print(f"  sin output: {len(sin_out)} {sin_out[:12]}")
    print(f"  errores: {len(errores)}")
    for i, name, val in errores:
        print(f"    [{i}] {name}: {val}")
    return len(errores) == 0
