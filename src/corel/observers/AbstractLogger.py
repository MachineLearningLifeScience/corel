class AbstractLogger:
    def initialize_logger(self, problem_setup_info, caller_info, seed, run_id=None):
        raise NotImplementedError("abstract method")

    def log_sequence(self, seq, step=None, verbose=False):
        raise NotImplementedError("abstract method")

    def log(self, d: dict, step: int, verbose=True):
        raise NotImplementedError("abstract method")

    def finish(self):
        raise NotImplementedError("abstract method")
