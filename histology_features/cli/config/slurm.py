from dataclasses import dataclass

@dataclass
class SLURMConfig:
    TIME: str
    CPU_PER_TASK: int
    MEMORY: str
    PARTITION: str = "standard"

    @classmethod
    def for_task(cls, task: str):
        if task.casefold() in [
            "align", "alignment", "registration", "submit_registration"
            ]:
            return cls(
                TIME="08:00:00", 
                CPU_PER_TASK=24,
                MEMORY="256G",

                )
        elif task == "concatenate":
            return cls(
                TIME="02:00:00", 
                CPU_PER_TASK=12,
                MEMORY="32G",
                )
        else:
            raise ValueError(f"Unknown task: {task}")