# main.py
import pygame
from game import Game

def main():
    game = Game(csv_path='./data/EURUSD_5m.csv')
    game_mode = 'human'  # 可扩展为agent模式

    if game_mode == 'human':
        game.run_human_mode()
    elif game_mode == 'agent':
        game.run_agent_mode(None)

if __name__ == "__main__":
    main()
