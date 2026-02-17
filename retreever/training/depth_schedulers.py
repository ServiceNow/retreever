import math
import random


# TODO: implement checkpointing for depth schedulers
class LinearDepthScheduler:
    def __init__(self, max_steps: int = 1000, min_value: int = 4, max_value: int = 11):
        """Gradually increase depth at which computing the loss during training warmup.

        Args:
            max_steps (int, optional): Total number of steps over which increasing depth. Defaults to 1000.
            min_value (int, optional): Initial depth value. Defaults to 4.
            max_value (int, optional): Final depth value. Defaults to 11.
        """
        self.depth = min_value - 1
        self.final_depth = max_value
        self.step_interval = math.ceil(max_steps / (max_value - min_value))

        if self.step_interval == 0:
            self.depth = self.final_depth

    def get_depth(self, global_step):
        if self.depth < self.final_depth and global_step % self.step_interval == 0:
            self.depth += 1

        return self.depth


class ExponentialDepthScheduler:
    def __init__(
        self,
        max_steps: int = 1000,
        min_value: int = 4,
        max_value: int = 11,
        multiply_factor: int = 2,
    ):
        """Gradually increase depth at which computing the loss during training warmup.

        Args:
            max_steps (int, optional): Total number of steps over which increasing depth. Defaults to 1000.
            min_value (int, optional): Initial depth value. Defaults to 4.
            max_value (int, optional): Final depth value. Defaults to 11.
            multiply_factor (int, optional): Each depth value will have <multiply_factor> times more iterations than the previous depth. Defaults to 2.
        """
        self.depth = min_value
        self.final_depth = max_value
        self.mul_factor = multiply_factor

        num_unit_intervals = multiply_factor ** (max_value - min_value + 1)
        self.unit_step_interval = math.ceil(max_steps / num_unit_intervals)
        self.current_step_interval = self.unit_step_interval
        self.last_change = 0

        if self.unit_step_interval == 0:
            self.depth = self.final_depth

    def get_depth(self, global_step):
        if (
            self.depth < self.final_depth
            and (global_step - self.last_change) == self.current_step_interval
        ):
            self.depth += 1  # increase depth
            self.current_step_interval *= (
                self.mul_factor
            )  # next change will be in multiply_factor * unit_interval steps
            self.last_change = global_step

        return self.depth


class RandomDepthScheduler:
    def __init__(self, min_value: int = 1, max_value: int = 10, *args, **kwargs):
        """Randomly choose depth at which computing the loss during training.
        Args:
            min_value (int, optional): Initial depth value. Defaults to 1.
            max_value (int, optional): Final depth value. Defaults to 10.
        """
        self.min_value = min_value
        self.max_value = max_value

    def get_depth(self, *args):
        # Depth chosen in such a way that the larger depths are chosen much more often.
        self.depth = random.choices(
            range(self.min_value, self.max_value + 1),
            weights=range(self.min_value, self.max_value + 1),
        )[0]
        return self.depth
    

class RandomHeavyTailedDepthScheduler:
    def __init__(self, min_value: int = 1, max_value: int = 10, *args, **kwargs):
        """Randomly choose depth at which computing the loss during training.
        Args:
            min_value (int, optional): Initial depth value. Defaults to 1.
            max_value (int, optional): Final depth value. Defaults to 10.
        """
        self.min_value = min_value
        self.max_value = max_value

    def get_depth(self, *args):
        # Depth chosen in such a way that the larger depths are chosen much more often.
        self.depth = random.choices(
            range(self.min_value, self.max_value + 1),
            weights=[x**2 for x in range(self.min_value, self.max_value + 1)],
        )[0]
        return self.depth
    
class RandomUniformDepthScheduler:
    def __init__(self, min_value: int = 1, max_value: int = 10, *args, **kwargs):
        """Randomly choose depth at which computing the loss during training.
        Args:
            min_value (int, optional): Initial depth value. Defaults to 1.
            max_value (int, optional): Final depth value. Defaults to 10.
        """
        self.min_value = min_value
        self.max_value = max_value

    def get_depth(self, *args):
        # Depth chosen in such a way that the larger depths are chosen much more often.
        self.depth = random.choices(
            range(self.min_value, self.max_value + 1),
        )[0]
        return self.depth


KNOWN_SCHEDULERS = {
    "linear": LinearDepthScheduler,
    "exponential": ExponentialDepthScheduler,
    "random": RandomHeavyTailedDepthScheduler,
    "random_uniform": RandomUniformDepthScheduler,
    "random_linear": RandomDepthScheduler,
}
