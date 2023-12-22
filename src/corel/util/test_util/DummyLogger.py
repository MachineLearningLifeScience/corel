from corel.observers.AbstractLogger import AbstractLogger


class DummyLogger(AbstractLogger):
    def initialize_logger(self, problem_setup_info, caller_info, seed, run_id=None):
        pass

    def log_sequence(self, seq, step=None, verbose=False):
        pass

    def log(self, d: dict, step: int, verbose=True):
        pass

    def finish(self):
        pass
