import pygame
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

# --- Parameters ---
WIDTH, HEIGHT = 800, 600
FPS = 60            # max frames per second (screen refresh)
dt_ms = 10          # timestep in milliseconds (content update rate)

# --- Init pygame ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Quota Game Demo")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 48)

# Setup
last_update = datetime.now()
counter = 0   

r = 0.01
s = 0.1474
S0 = 100
m = r - 0.5*s*s
dt = 1/365/24/60/60/100 #10 ms
n = 100*60*10

def geom_brownian_motion(S0, m, s, n, dt):
    logret = np.random.normal(m * dt, s * np.sqrt(dt), size=int(n))
    ret = np.exp(logret.cumsum())
    price = S0*ret
    return price

t = pd.date_range(start=datetime.now(), periods=n, freq="10ms")
S = geom_brownian_motion(S0, m, s, n, dt)

# --- Display parameters ---
anchor = (WIDTH // 4, HEIGHT // 2)  # fixed anchor point for the most recent value
scale_y = 50000.0                       # pixels per price unit
spacing_x = 20                      # horizontal spacing between steps


plt.plot(t, S)

# --- Main loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # check if dt has passed
    now = datetime.now()
    if (now - last_update) >= timedelta(milliseconds=dt_ms):
        counter += 1              # update game state
        last_update = now         # reset timer

    # --- Clear screen each frame ---
    screen.fill((30, 30, 30))

    if counter > 0:
        # take last 20 values
        window = S[max(0, counter-20):counter+1]
        latest = window[-1]

        # draw polyline
        points = []
        for j, val in enumerate(window[::-1]):  # reversed, newest on the right
            x = anchor[0] - j * spacing_x
            y = anchor[1] - (val - latest) * scale_y
            points.append((x, y))

        if len(points) > 1:
            pygame.draw.lines(screen, (0, 200, 100), False, points, 2)

        # draw anchor point
        pygame.draw.circle(screen, (200, 50, 50), anchor, 5)

        # text for latest value
        text = font.render(f"{latest:.5f}", True, (200, 200, 200))
        screen.blit(text, (anchor[0] + 10, anchor[1] - 20))

    pygame.display.flip()
    clock.tick(FPS)   # keep loop from eating CPU

pygame.quit()
sys.exit()
