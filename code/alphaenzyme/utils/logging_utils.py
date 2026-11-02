import logging.config
import time
from logging import BASIC_FORMAT
import torch


def singleton(cls):
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return get_instance()


@singleton
class LoggerWrapper(object):
    def __init__(self):
        self.basic_format = "%(asctime)s %(levelname)s %(process)d [%(filename)s:%(lineno)d %(funcName)s] %(message)s"
        logging.basicConfig(
            format=self.basic_format,
            level=logging.INFO,
        )
        self.logger = logging.getLogger("root")
        self.rank = 0
        self.verbose_level = 0

    def set_logger_from_configs(self, configs):
        # Maybe something should be set.
        if configs is None:
            return
        
        for k, v in configs.items():
            if k == 'env':
                self.set_logger(env=v)
            else:
                setattr(self, k, v)


    def set_logger(self, env="terminal"):
        if env == "terminal":
            logging.basicConfig(
                format="%(asctime)s %(levelname)s %(process)d [%(filename)s:%(lineno)d %(funcName)s] %(message)s",
                level=logging.INFO,
            )
            self.logger = logging.getLogger("root")
        else:
            self.logger = logging.getLogger("root")
            self.logger.setLevel(logging.INFO)

            BASIC_FORMAT = "%(asctime)s %(levelname)s %(process)d [%(filename)s:%(lineno)d %(funcName)s] %(message)s"
            formatter = logging.Formatter(BASIC_FORMAT)

            chlr = logging.StreamHandler()
            chlr.setLevel(logging.INFO)
            chlr.setFormatter(formatter)

            file_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".log"
            fhlr = logging.FileHandler(file_name)
            fhlr.setLevel(logging.INFO)
            fhlr.setFormatter(formatter)

            self.logger.addHandler(chlr)
            self.logger.addHandler(fhlr)

    def set_rank(self, rank):
        self.rank = rank

    def info(self, msg, verbose_level=2, ignore_rank = False, log_rank=True):
        if log_rank:
            msg = f"Rank {self.rank}: {msg}"

        if ignore_rank:
            if verbose_level > self.verbose_level:
                self.logger.info(msg, stacklevel=2)
        else:
            if self.rank == 0:
                if verbose_level > self.verbose_level:
                    self.logger.info(msg, stacklevel=2)

    def warning(self, msg, verbose_level=2, log_rank=True):
        if log_rank:
            msg = f"Rank {self.rank}: {msg}"
        if verbose_level > self.verbose_level:
            self.logger.warning(msg, stacklevel=2)

    def debug(self, msg):
        self.logger.debug(msg, stacklevel=2)

    def error(self, msg):
        self.logger.error(msg, stacklevel=2)

    @property
    def seperator(self):
        return "<" * 30

    def line(self, num=100):
        return "-" * num

    def report_evaluate_metrics(self, summary_writer, prefix, idx, loss, step, avg_metrics, mode='val', rank=0):
        # From BM scoring, to report the metric results of one evaluation.
        # if rank == 0 part is modified to suit our framework.
        string = f"the {idx} evaluation loss at {prefix} | "
        string += "loss: {:.6E} | ".format(loss)
        for key in avg_metrics:
            string += " {} {:.6E} | ".format(key, avg_metrics[key])

        string += f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB"
        string += f"Max GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB"
        length = len(string) + 1
        if rank == 0:
            self.info(self.separator)
            self.info(self.line(length))
            self.info(string)
            self.info(self.line(length))
        if summary_writer is not None:
            log_dict = {f"{mode}_{idx}/{k}": v for k, v in avg_metrics.items()}
            log_dict[f"{mode}_{idx}/loss"] = loss
            summary_writer.log_scalar_dict(log_dict, step)

    def log_train_step(self, record, step_id, category="train/batch"):
        # From TorchDrug, to report one batch records?
        # May be deprecated future
        if category.endswith("batch"):
            self.logger.warning(self.separator)
        elif category.endswith("epoch"):
            self.logger.warning(self.line)
        if category == "train/epoch":
            for k in sorted(record.keys()):
                self.logger.warning("average %s: %g" % (k, record[k]))
        else:
            for k in sorted(record.keys()):
                self.logger.warning("%s: %g" % (k, record[k]))

    # def report_single_eval_case(self, summary_writer, prefix, idx, loss, step, per_sample_metrics, mode='val', rank=0):
    #     # 可通过开关，选择是否打印每一个sample的evaluation metrics
    #     pass