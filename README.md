# 🐍 SnakeAI

![Python](https://img.shields.io/badge/python-3.13-blue?logo=python)
![PyTorch](https://img.shields.io/badge/pytorch-%23EE4C2C?logo=pytorch\&logoColor=white)
![Pygame](https://img.shields.io/badge/pygame-2.1-000?logo=pygame)

Нейросетевой агент, обучающийся играть в «Змейку» (Pygame + простая Linear Q-Net). Включает игру, модель, тренер и визуализацию прогресса.

## Быстрый запуск

```bash
git clone https://github.com/SirRigterion/SnakeAI.git
cd <repo-folder>

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
python agent.py
```

По умолчанию `agent.py` запускает игру с отрисовкой (`render=True`). Для быстрой тренировки без окна установите `game = SnakeGameAI(render=False)` в `agent.py`.

## Как загрузить модель (коротко)

В `model.py` есть `load(file_name='model.pth')`. Пример использования:

```python
from model import Linear_QNet
from game import SnakeGameAI
from agent import Agent
import torch

agent = Agent()
agent.model.load('model.pth')   # ищет ./model/model.pth
game = SnakeGameAI(render=True)
# далее: использовать model для выбора действий (см. agent.get_action)
```

## Важные параметры

* `BLOCK_SIZE` (game.py) — размер клетки в пикселях (по умолчанию 10)
* `SPEED` (game.py) — fps визуализации (по умолчанию 60)
* `LR`, `gamma`, `MAX_MEMORY`, `BASIC_SIZE` (agent.py) — гиперпараметры обучения
* `render` в `SnakeGameAI` — `True` = окно + график, `False` = headless (быстрее)

## Известные ограничения

* Агент требует длительного обучения, по умолчанию не оптимален и часто ходит бессмысленно, но визуально за частую это выглядит будто змея обвивает свою добычу.
* Встроенные shaping-награды помогают, но не гарантируют быстрый успех — для результатов нужен большой объём эпизодов и/или тюнинг гиперпараметров.
* BFS-проверки и отрисовка заметно замедляют обучение — для ускорения используйте `render=False` и/или уменьшите частоту вычислений графика.