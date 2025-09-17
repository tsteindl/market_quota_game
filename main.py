import pygame
import numpy as np
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

def gbm_step(S0, m, s, dt):
    logret = np.random.normal(m * dt, s * np.sqrt(dt))
    ret = np.exp(logret)
    price = S0*ret
    return price

# --- Display parameters ---
anchor = (WIDTH // 4, HEIGHT // 2)
scale_y = 50000.0
spacing_x = 1
window_length = 200

# first value line
S = [S0]
first_value = S0

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
rect_final_val = None

released = False
paused = False
release_counter = None
release_value = None
hit = False
hit_point = None
hit_point_val = None

budget = 10000
stake = 1000
stake_min = 100
stake_max = 5000
hit_processed = False  # to ensure we update budget only once

# --- Main loop ---
running = True
while running:
    mouse_pos = pygame.mouse.get_pos()
    mouse_pressed = pygame.mouse.get_pressed()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEWHEEL:
            # scroll up → zoom in, scroll down → zoom out
            if event.y > 0: 
                scale_y *= 1.1   # zoom in
            else:
                scale_y /= 1.1   # zoom out
            scale_y = max(1000, min(scale_y, 200000))
        elif event.type == pygame.KEYDOWN:
            # adjust stake with up/down arrows
            if event.key == pygame.K_UP:
                stake = min(stake_max, stake + 1000)
            elif event.key == pygame.K_DOWN:
                stake = max(stake_min, stake - 1000)
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
                
                rect_final_val = {
                    "x_left": rect_final.left,
                    "x_width": rect_final.width,
                    "y_top_val": release_value - (anchor[1] - rect_final.top)/scale_y,
                    "y_bottom_val": release_value - (anchor[1] - rect_final.bottom)/scale_y
                }
            else:
                rect_start = rect_end = rect_final = None  # discard

    # update counter only if not paused
    now = datetime.now()
    if not paused and (now - last_update) >= timedelta(milliseconds=dt_ms):
        S_new = gbm_step(S[-1], m, s, dt)
        S.append(S_new)
        
        counter += 1
        last_update = now

    # --- Clear ---
    screen.fill((30, 30, 30))

    # --- Draw budget and stake ---
    budget_text = font.render(f"Budget: ${budget}", True, (255, 255, 0))
    screen.blit(budget_text, (10, 10))
    stake_text = font.render(f"Stake: ${stake} (UP/DOWN to change)", True, (255, 255, 255))
    screen.blit(stake_text, (10, 30))


    if counter > 0:
        # past history up to current counter
        if released:
            past_window = S[max(0, release_counter-window_length):release_counter+1]
        else:
            past_window = S[max(0, counter-window_length):counter+1]
        past_latest = past_window[-1]

        past_points = []
        for j, val in enumerate(past_window[::-1]):
            if released:
                x = anchor[0] - j * spacing_x
                y = anchor[1] - (val - past_latest) * scale_y
            else:
                x = anchor[0] - j * spacing_x
                y = anchor[1] - (val - past_latest) * scale_y
            past_points.append((x, y))

        if len(past_points) > 1:
            pygame.draw.lines(screen, (0, 200, 100), False, past_points, 2)

        # released forward path
        if released:
            forward_window = S[release_counter:counter+1]
            forward_points = [(anchor[0] + j * spacing_x,
                            anchor[1] - (val - release_value) * scale_y)
                            for j, val in enumerate(forward_window)]

            if len(forward_points) > 1:
                pygame.draw.lines(screen, (0, 200, 100), False, forward_points, 2)

            # check hit only once
            if rect_final_val and not hit:
                for j, (px, py) in enumerate(forward_points):
                    p_val = release_value - (anchor[1] - py)/scale_y
                    if (rect_final_val["x_left"] <= px <= rect_final_val["x_left"] + rect_final_val["x_width"] and 
                        rect_final_val["y_top_val"] <= p_val <= rect_final_val["y_bottom_val"]):
                        hit = True
                        hit_point_val = p_val
                        hit_point_x = px
                        paused = True
                        break

            # pause if last forward point reaches rectangle right edge
            if rect_final_val and forward_points[-1][0] >= rect_final_val["x_left"] + rect_final_val["x_width"]:
                paused = True

        # anchor marker
        pygame.draw.circle(screen, (200, 50, 50), anchor, 5)
        
        draw_dashed_line(screen, (200, 50, 50),
                         (anchor[0], 0), (anchor[0], HEIGHT))
        
        # latest value text
        latest = S[counter]

        if released and len(forward_points) > 0:
            px, py = forward_points[-1]
        else:
            px, py = past_points[0]

        text = font.render(f"{latest:.5f}", True, (200, 200, 200))
        screen.blit(text, (px + 10, py - 20))

        # dashed horizontal line at first value
        ref_val = release_value if released else past_latest
        y_first = anchor[1] - (first_value - ref_val) * scale_y
        draw_dashed_line(screen, (200, 200, 0), (0, int(y_first)), (WIDTH, int(y_first)), width=1)
        
        y_first_text = font.render(f"{first_value:.5f}", True, (200, 200, 200))
        screen.blit(y_first_text, (WIDTH - 70, y_first - 20))

        # dashed band for min offset
        draw_dashed_line(screen, (100, 100, 100),
                         (anchor[0]+min_offset, 0), (anchor[0]+min_offset, HEIGHT))

        # draw rectangle
        if rect_final_val:
            x_left = rect_final_val["x_left"]
            width = rect_final_val["x_width"]
            y_top = anchor[1] - (release_value - rect_final_val["y_top_val"]) * scale_y
            y_bottom = anchor[1] - (release_value - rect_final_val["y_bottom_val"]) * scale_y
            height = y_bottom - y_top
            color = (50, 200, 50) if hit else (50, 50, 200)
            pygame.draw.rect(screen, color, (x_left, y_top, width, height), 2)
            
            # --- Draw multiplier text ---
            multiplier = 2
            multiplier_text = font.render(f"x{multiplier}", True, (255, 255, 0))
            text_rect = multiplier_text.get_rect(center=(x_left + width/2, y_top + height/2))
            screen.blit(multiplier_text, text_rect)
            
            # --- Update budget when hit/miss processed ---
            if paused and not hit_processed:
                if hit:
                    budget += stake * 2
                else:
                    budget -= stake
                hit_processed = True
            
            
        elif rect_start and rect_end:
            rect_temp = pygame.Rect(min(rect_start[0], rect_end[0]),
                                    min(rect_start[1], rect_end[1]),
                                    abs(rect_end[0]-rect_start[0]),
                                    abs(rect_end[1]-rect_start[1]))
            pygame.draw.rect(screen, (100, 100, 200), rect_temp, 1)
            
        if hit and hit_point_val is not None:
            py = anchor[1] - (release_value - hit_point_val) * scale_y
            pygame.draw.circle(screen, (255, 0, 0), (int(hit_point_x), int(py)), 6)
    
    # --- Draw resume button if paused ---
    button_rect = pygame.Rect(10, 50, 100, 30)
    if paused:
        pygame.draw.rect(screen, (200, 200, 50), button_rect)
        button_text = font.render("Resume", True, (0, 0, 0))
        screen.blit(button_text, (button_rect.x + 10, button_rect.y + 5))
    
    # --- Handle mouse click on resume button ---
    if paused and mouse_pressed[0]:  # left mouse button
        if button_rect.collidepoint(mouse_pos):
            paused = False
            rect_start = rect_end = rect_final_val = None
            released = False
            hit = False
            hit_processed = False  # reset for next round
            
            first_value = S[counter]

            # Recalculate S starting from current value
            S0_new = S[counter]
            release_value = S0_new  # new baseline for next rectangle
            n_remaining = n  # or however many points you want for the next forward simulation
            counter = 0  # reset counter for new forward path

    pygame.display.flip()
    clock.tick(FPS)