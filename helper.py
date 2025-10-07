import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pygame
import numpy as np


def plot(scores, mean_scores):
    dpi = 100
    fig, ax = plt.subplots(figsize=(4, 4.8), dpi=dpi)
    ax.set_title('Обучение')
    ax.set_xlabel('Номер игры')
    ax.set_ylabel('Счёт')
    if len(scores) > 0:
        ax.plot(scores, label='Счёт')
    if len(mean_scores) > 0:
        ax.plot(mean_scores, label='Средний счёт')
    ax.set_ylim(ymin=0)
    if len(scores) > 0:
        try:
            ax.text(len(scores)-1, scores[-1], f"{scores[-1]:.1f}")
        except Exception:
            pass
    if len(mean_scores) > 0:
        try:
            ax.text(len(mean_scores)-1, mean_scores[-1], f"{mean_scores[-1]:.1f}")
        except Exception:
            pass
    ax.legend()
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    buf = np.asarray(renderer.buffer_rgba())
    raw_data = buf[:, :, :3].tobytes()
    size = fig.canvas.get_width_height()
    surf = pygame.image.fromstring(raw_data, size, "RGB")
    plt.close(fig)
    return surf
