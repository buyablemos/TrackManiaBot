import multiprocessing

class Events:
    def __init__(self):
        self.choose_map_event = multiprocessing.Event()
        self.print_state_event = multiprocessing.Event()
        self.load_model_event = multiprocessing.Event()
        self.save_model_event = multiprocessing.Event()
        self.quit_event = multiprocessing.Event()
        self.embed_game_event = multiprocessing.Event()