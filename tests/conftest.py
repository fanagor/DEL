"""A strict in-memory W&B service; tests never send experiment data remotely."""

import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


class FakeWandb:
    Settings = SimpleNamespace
    Table = staticmethod(lambda **kwargs: kwargs)
    Image = staticmethod(lambda pixels, **kwargs: {"pixels": pixels, **kwargs})

    def __init__(self):
        self.runs = {}
        self.starts = []
        self.finishes = []
        self.rewinds = []
        self.accept_logs = True
        self.reject_rewind = False
        self.before_log = None
        self.after_log = None
        self.before_init = None

    def init(self, **kwargs):
        self.starts.append(kwargs)
        if self.before_init:
            self.before_init(kwargs)
        identifier = kwargs["id"]
        entity = kwargs["entity"] or "default-team"
        if kwargs.get("resume") == "must" and identifier not in self.runs:
            raise RuntimeError("Remote run missing")
        if "resume_from" in kwargs:
            assert "resume" not in kwargs
            if self.reject_rewind:
                raise RuntimeError("Rewind unavailable")
            previous, point = kwargs["resume_from"].split("?_step=")
            assert previous == identifier
            assert int(point) >= 0
            self.rewinds.append((identifier, int(point)))
            self.runs[identifier]["rows"] = self.runs[identifier]["rows"][: int(point) + 1]
        record = self.runs.setdefault(
            identifier, {"rows": [], "summary": {}, "entity": entity, "project": kwargs["project"]}
        )
        assert record["entity"] == entity
        assert record["project"] == kwargs["project"]
        # Simulate libraries consuming randomness during startup and log calls.
        random.random()
        np.random.random()
        torch.rand(1)
        service = self

        class Run:
            id = identifier
            project = record["project"]
            summary = record["summary"]

            def __init__(self):
                self.entity = entity
                self.step = len(record["rows"])

            def define_metric(self, *args, **kwargs):
                pass

            def log(self, row, *, step, commit):
                assert commit is True
                assert step == self.step, "Duplicate or missing W&B history position"
                if service.before_log:
                    service.before_log(self, row, step)
                random.random()
                np.random.random()
                torch.rand(1)
                self.step += 1
                if service.accept_logs:
                    assert len(record["rows"]) == step
                    record["rows"].append(dict(row, _step=step))
                    record["summary"].update(row)
                if service.after_log:
                    service.after_log(self, row, step)

            def finish(self, exit_code=0):
                service.finishes.append((identifier, exit_code))

        return Run()


@pytest.fixture
def fake_wandb(monkeypatch):
    service = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", service)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    return service
