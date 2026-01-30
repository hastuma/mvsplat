import os
import csv
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only

LOG_PATH = Path("outputs/local")


class LocalLogger(Logger):
    def __init__(self, output_dir: Optional[Path] = None) -> None:
        super().__init__()
        self.experiment = None
        # Set up CSV logging
        self.output_dir = output_dir if output_dir is not None else LOG_PATH
        self.metrics_file = self.output_dir / "metrics.csv"
        self.metrics_file.parent.mkdir(exist_ok=True, parents=True)
        self._csv_writer = None
        self._csv_file = None
        self._fieldnames = set()

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # Write metrics to CSV file
        if not metrics:
            return
        
        # Add step to metrics
        metrics_with_step = {"step": step, **metrics}
        
        # Check if we need to initialize or update CSV headers
        new_fields = set(metrics_with_step.keys()) - self._fieldnames
        if new_fields or self._csv_writer is None:
            self._fieldnames.update(metrics_with_step.keys())
            
            # Read existing data if file exists
            existing_data = []
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    reader = csv.DictReader(f)
                    existing_data = list(reader)
            
            # Close old file if open
            if self._csv_file is not None:
                self._csv_file.close()
            
            # Rewrite with new headers
            self._csv_file = open(self.metrics_file, 'w', newline='')
            self._csv_writer = csv.DictWriter(
                self._csv_file, 
                fieldnames=sorted(self._fieldnames)
            )
            self._csv_writer.writeheader()
            
            # Write back existing data
            for row in existing_data:
                self._csv_writer.writerow(row)
        
        # Write the new metrics
        self._csv_writer.writerow(metrics_with_step)
        self._csv_file.flush()

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        captions = kwargs.get("caption", None)
        
        for index, image in enumerate(images):
            name_part = f"{index:0>2}"
            if captions is not None:
                if isinstance(captions, list) and index < len(captions):
                     name_part = f"{captions[index]}"
                elif isinstance(captions, str):
                     name_part = f"{captions}"
                     if len(images) > 1:
                         name_part += f"_{index}"
            
            # Sanitize filename
            if isinstance(name_part, str):
                name_part = "".join(c for c in name_part if c.isalnum() or c in ('_', '-'))

            path = LOG_PATH / f"{key}/{step:0>6}_{name_part}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(image).save(path)
