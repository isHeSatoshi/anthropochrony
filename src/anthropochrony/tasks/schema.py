from dataclasses import asdict, dataclass, field


@dataclass
class Task:
    task_id: str
    family: str
    source: str
    source_revision: str
    license: str
    task_turn: str
    harness: str
    entry_point: str | None = None
    reference_solution: str = ""
    test_code: str = ""
    io_examples: dict | None = None
    public_examples: dict = field(default_factory=dict)
    split: str = "unassigned"
    difficulty: str = ""
    near_dup_group: str = ""
    tests_ref: str = ""
    ref_tokens: int = 0
    prompt_tokens: int = 0
    time_limits: dict = field(default_factory=lambda: {"per_test_s": 3, "per_task_s": 60})
    gt: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)

    def as_dict(self):
        if self.harness not in {"evalplus_check", "unittest", "lcd_check", "stdio", "pytest"}:
            raise ValueError(f"Unknown task harness {self.harness}")
        if not self.task_id or not self.source_revision or not self.license:
            raise ValueError("Task ID, source revision, and per-row license are required")
        return asdict(self)
