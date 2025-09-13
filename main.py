import pygame
import sys

# initialize pygame
pygame.init()

# window settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Financial Market Quota Game")

# main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill background
    screen.fill((30, 30, 30))

    # draw text
    font = pygame.font.SysFont(None, 48)
    text = font.render("Hello Markets!", True, (200, 200, 200))
    screen.blit(text, (250, 280))

    # update display
    pygame.display.flip()

pygame.quit()
sys.exit()
