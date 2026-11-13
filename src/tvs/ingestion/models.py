from dataclasses import dataclass


@dataclass
class IngestionStats:
    frames_discovered: int = 0
    frames_processed: int = 0
    frames_skipped: int = 0

    @property
    def skip_ratio(self) -> float:
        if self.frames_discovered == 0:
            return 0.0

        return self.frames_skipped / self.frames_discovered
