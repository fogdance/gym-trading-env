# main.py
from game import Game
from utils import load_csv_data

def main():
    game = Game(load_csv_data('./data/EURUSD_5m.csv'))
    game_mode = 'human'  # 可扩展为agent模式

    if game_mode == 'human':
        game.run_human_mode()
    elif game_mode == 'agent':
        game.run_agent_mode(None)

if __name__ == "__main__":
    main()
