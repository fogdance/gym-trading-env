# src/gym_trading_env/rendering/info_logging_callback.py

from stable_baselines3.common.callbacks import BaseCallback
import wandb
from torch.utils.tensorboard import SummaryWriter
from decimal import Decimal

class InfoLoggingCallback(BaseCallback):
    """
    Custom callback to extract the info dictionary from the callback's locals
    and log it to WandB and TensorBoard.
    """

    def __init__(self, log_dir, verbose=0):
        super(InfoLoggingCallback, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir=log_dir)
        # Optionally log key metrics to avoid performance issues with TensorBoard
        self.important_keys = ['balance', 'realized_pnl', 'broker_balance', 'fees_collected', 'used_margin']

    def _on_step(self) -> bool:
        """
        Executed after each training step to access the current step's info.
        """
        # Get the info from the current step
        infos = self.locals.get('infos', [])
        if infos:
            for info in infos:
                # Convert Decimal to float
                info_float = {k: float(v) if isinstance(v, Decimal) else v for k, v in info.items()}

                # Log to WandB
                wandb.log(info_float, step=self.num_timesteps)

                for key, value in info_float.items():
                    # Log to TensorBoard (optionally filter by important keys)
                    if key in self.important_keys:
                        self.writer.add_scalar(key, value, self.num_timesteps)

        return True  # Continue training

    def _on_training_end(self):
        # Close the SummaryWriter
        self.writer.close()
        print(f"Training ended. Total steps: {self.num_timesteps}.")
