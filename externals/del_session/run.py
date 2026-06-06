"""$del_session — delete this run's session folder after the script finishes."""

from __future__ import annotations

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    ctx.session.delete_after_run = True
    return inp.bundle.copy()
