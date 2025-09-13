import pygame
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

# --- Parameters ---
WIDTH, HEIGHT = 800, 600
FPS = 60
dt_ms = 10

# --- Init pygame ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Quota Game Demo")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)   # smaller text

# Setup
last_update = datetime.now()
counter = 0

r = 0.01
s = 0.1474
S0 = 100
m = r - 0.5*s*s
dt = 1/365/24/60/60/100 #10 ms in years
n = 100*60*10

def geom_brownian_motion(S0, m, s, n, dt):
    logret = np.random.normal(m * dt, s * np.sqrt(dt), size=int(n))
    ret = np.exp(logret.cumsum())
    price = S0*ret
    return price

t = pd.date_range(start=datetime.now(), periods=n, freq="10ms")
S = geom_brownian_motion(S0, m, s, n, dt)

# --- Display parameters ---
anchor = (WIDTH // 4, HEIGHT // 2)
scale_y = 50000.0
spacing_x = 1
window_length = 200

# first value line
first_value = S[0]

# --- Mouse interaction ---
dragging = False
rect_start = None
rect_end = None
min_offset = 200  # rectangle must be this far from anchor horiz

def draw_dashed_line(surf, color, start, end, width=1, dash_length=20):
    x1, y1 = start
    x2, y2 = end
    dl = dash_length
    if x1 == x2:  # vertical
        ycoords = range(y1, y2, dl if y2>y1 else -dl)
        for y in ycoords:
            pygame.draw.line(surf, color, (x1, y), (x2, min(y+dl, y2)), width)
    else:         # horizontal
        xcoords = range(x1, x2, dl if x2>x1 else -dl)
        for x in xcoords:
            pygame.draw.line(surf, color, (x, y1), (min(x+dl, x2), y2), width)

# --- State variables ---
dragging = False
rect_start = None
rect_end = None
rect_final = None

released = False
paused = False
release_counter = None
release_value = None
hit = False


# --- Main loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and not released:
            if event.button == 1:
                dragging = True
                rect_start = event.pos
                rect_end = event.pos
        elif event.type == pygame.MOUSEMOTION and dragging:
            rect_end = event.pos
        elif event.type == pygame.MOUSEBUTTONUP and dragging:
            dragging = False
            rect_end = event.pos
            rect_temp = pygame.Rect(min(rect_start[0], rect_end[0]),
                                    min(rect_start[1], rect_end[1]),
                                    abs(rect_end[0]-rect_start[0]),
                                    abs(rect_end[1]-rect_start[1]))
            # enforce vertical distance
            if rect_temp.centerx - anchor[0] >= min_offset:
                rect_final = rect_temp
                released = True
                release_counter = counter
                release_value = S[counter]
            else:
                rect_start = rect_end = rect_final = None  # discard

    # update counter only if not paused
    now = datetime.now()
    if not paused and (now - last_update) >= timedelta(milliseconds=dt_ms):
        counter += 1
        last_update = now

    # --- Clear ---
    screen.fill((30, 30, 30))

    if counter > 0:
        # past history up to current counter
        past_window = S[max(0, counter-window_length):counter+1]
        past_latest = past_window[-1]

        past_points = []
        for j, val in enumerate(past_window[::-1]):
            x = anchor[0] - j * spacing_x
            y = anchor[1] - (val - past_latest) * scale_y
            past_points.append((x, y))

        if len(past_points) > 1:
            pygame.draw.lines(screen, (0, 200, 100), False, past_points, 2)

        # released forward path
        if released:
            forward_window = S[release_counter:counter+1]
            forward_points = []
            for j, val in enumerate(forward_window):
                x = anchor[0] + j * spacing_x
                y = anchor[1] - (val - release_value) * scale_y
                forward_points.append((x, y))

            if len(forward_points) > 1:
                pygame.draw.lines(screen, (0, 150, 250), False, forward_points, 2)

            # check hit
            if rect_final and not hit:
                for p in forward_points:
                    if rect_final.collidepoint(p):
                        hit = True
                        paused = True
                        break

            # pause when path reaches rectangle right edge
            if rect_final and forward_points[-1][0] >= rect_final.right:
                paused = True

        # anchor marker
        pygame.draw.circle(screen, (200, 50, 50), anchor, 5)

        # latest value text
        latest = S[counter]
        text = font.render(f"{latest:.5f}", True, (200, 200, 200))
        screen.blit(text, (past_points[0][0] + 10, past_points[0][1] - 20))

        # dashed horizontal line at first value
        ref_val = release_value if released else past_latest
        y_first = anchor[1] - (first_value - ref_val) * scale_y
        draw_dashed_line(screen, (200, 200, 0), (0, int(y_first)), (WIDTH, int(y_first)), width=1)

        # dashed band for min offset
        draw_dashed_line(screen, (100, 100, 100),
                         (anchor[0]+min_offset, 0), (anchor[0]+min_offset, HEIGHT))

        # draw rectangle
        if rect_final:
            color = (50, 200, 50) if hit else (50, 50, 200)
            pygame.draw.rect(screen, color, rect_final, 2)
        elif rect_start and rect_end:
            rect_temp = pygame.Rect(min(rect_start[0], rect_end[0]),
                                    min(rect_start[1], rect_end[1]),
                                    abs(rect_end[0]-rect_start[0]),
                                    abs(rect_end[1]-rect_start[1]))
            pygame.draw.rect(screen, (100, 100, 200), rect_temp, 1)

    pygame.display.flip()
    clock.tick(FPS)