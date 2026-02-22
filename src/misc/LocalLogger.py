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
            existing_data = []
            if self.metrics_file.exists():
                with open(self.metrics_file, "r") as f:
                    reader = csv.DictReader(f)
                    existing_data = list(reader)
            if self._csv_file is not None:
                self._csv_file.close()

            self._csv_file = open(self.metrics_file, "w", newline="")
            self.sorted_fields = sorted(self._fieldnames)
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self.sorted_fields
            )
            self._csv_writer.writeheader()
            for row in existing_data:
                self._csv_writer.writerow(row)

            # Initialize or refresh text file with headers
            self.metrics_txt = self.output_dir / "metrics.txt"
            with open(self.metrics_txt, "w") as f:
                header_str = " | ".join(
                    f"{str(field):>15}" for field in self.sorted_fields
                )
                f.write(header_str + "\n")
                f.write("-" * len(header_str) + "\n")

        # Write CSV
        self._csv_writer.writerow(metrics_with_step)
        self._csv_file.flush()

        # Write Pretty Text
        with open(self.output_dir / "metrics.txt", "a") as f:

            def format_val(v):
                if isinstance(v, (int, float)):
                    return f"{v:15.6f}"
                return f"{str(v):>15}"

            row_str = " | ".join(
                format_val(metrics_with_step.get(f, "")) for f in self.sorted_fields
            )
            f.write(row_str + "\n")

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

            path = self.output_dir / f"{key}/{step:0>6}_{name_part}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(image).save(path)
