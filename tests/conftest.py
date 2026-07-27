"""
conftest.py — Configuração partilhada dos testes (pytest).

Adiciona a pasta src/ ao caminho de importação, para os testes poderem
fazer `from predict import ...` tal como a app faz.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
