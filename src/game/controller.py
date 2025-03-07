import pygame

class Controller:
    @staticmethod
    def get_human_action():
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            return 'left'
        elif keys[pygame.K_RIGHT]:
            return 'right'
        elif keys[pygame.K_SPACE]:
            return 'close'
        else:
            return 'hold'
