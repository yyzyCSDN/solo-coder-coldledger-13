from __future__ import annotations
from dataclasses import dataclass, replace
from decimal import Decimal
from threading import RLock
from typing import Iterable

from .network_allocation import PlannedMove, Plan, ZERO


def _d(value) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


@dataclass(frozen=True)
class StockRecord:
    """Versioned on-hand balance for (site, sku)."""
    site: str
    sku: str
    available: Decimal
    version: int = 0


@dataclass(frozen=True)
class LaneRecord:
    """Versioned remaining capacity for one lane."""
    source: str
    target: str
    available: Decimal
    sku: str | None = None
    version: int = 0

    @property
    def key(self) -> tuple[str, str, str | None]:
        return (self.source, self.target, self.sku)


@dataclass(frozen=True)
class WindowRecord:
    """Versioned remaining receiving capacity for one receiving window."""
    window_id: str
    site: str
    available: Decimal
    version: int = 0


@dataclass(frozen=True)
class Snapshot:
    """Optimistic-concurrency versions captured when the plan was built.
    Any row that changed between planning and commit aborts the commit."""
    stock: dict[tuple[str, str], int]
    lanes: dict[tuple[str, str, str | None], int]
    windows: dict[str, int]


@dataclass(frozen=True)
class CommitReceipt:
    reference: str
    stock: tuple[StockRecord, ...]
    lanes: tuple[LaneRecord, ...]
    windows: tuple[WindowRecord, ...]
    moves: tuple[PlannedMove, ...]


class StockBook:
    def __init__(self, rows: Iterable[StockRecord]):
        self.rows: dict[tuple[str, str], StockRecord] = {}
        for r in rows:
            if (r.site, r.sku) in self.rows:
                raise ValueError(f"duplicate stock row {r.site}/{r.sku}")
            self.rows[(r.site, r.sku)] = replace(r, available=_d(r.available))
        self.lock = RLock()


class LaneBook:
    def __init__(self, rows: Iterable[LaneRecord]):
        self.rows: dict[tuple[str, str, str | None], LaneRecord] = {}
        for r in rows:
            if r.key in self.rows:
                raise ValueError(f"duplicate lane row {r.key}")
            self.rows[r.key] = replace(r, available=_d(r.available))
        self.lock = RLock()


class WindowBook:
    def __init__(self, rows: Iterable[WindowRecord]):
        rows = list(rows)
        self.rows: dict[str, WindowRecord] = {
            r.window_id: replace(r, available=_d(r.available)) for r in rows
        }
        if len(self.rows) != len(rows):
            raise ValueError("duplicate window row")
        self.lock = RLock()


def resolve_lane(book: LaneBook, source: str, target: str, sku: str) -> LaneRecord:
    """Prefer an sku-specific lane record over a generic one."""
    exact = book.rows.get((source, target, sku))
    if exact is not None:
        return exact
    generic = book.rows.get((source, target, None))
    if generic is not None:
        return generic
    raise KeyError(f"unknown lane {source}->{target} {sku}")


def window_id(plan_obj: Plan, move: PlannedMove) -> str:
    ids = getattr(plan_obj, "window_ids", None)
    if ids is not None and move.window_index in ids:
        return ids[move.window_index]
    return f"window-{move.window_index}"


class TransferCommitter:
    """Commits a globally planned set of moves atomically across the three
    resource books: stock balances, lane capacity and receiving windows.

    All validation (idempotency, versions, availability, unknown keys,
    window-site consistency) runs before any row is mutated. Locks are
    always taken in one fixed order (stock -> lane -> window) so concurrent
    commits cannot deadlock; if any check fails nothing is written, which
    is the rollback.
    """

    def __init__(self, stock: StockBook, lanes: LaneBook, windows: WindowBook):
        self.stock = stock
        self.lanes = lanes
        self.windows = windows
        self._committed: set[str] = set()
        self.lock = RLock()

    def snapshot(self, plan_obj: Plan,
                 window_ids: dict[int, str] | None = None) -> Snapshot:
        """Read versions for every row this plan touches."""
        if window_ids is not None:
            plan_obj = _plan_with_window_ids(plan_obj, window_ids)
        stock_versions: dict[tuple[str, str], int] = {}
        lane_versions: dict[tuple[str, str, str | None], int] = {}
        window_versions: dict[str, int] = {}
        with self.stock.lock, self.lanes.lock, self.windows.lock:
            for m in plan_obj.moves:
                stock_versions[(m.source, m.sku)] = self.stock.rows[(m.source, m.sku)].version
                stock_versions[(m.target, m.sku)] = self.stock.rows[(m.target, m.sku)].version
                lane = resolve_lane(self.lanes, m.source, m.target, m.sku)
                lane_versions[lane.key] = lane.version
                wid = window_id(plan_obj, m)
                window_versions[wid] = self.windows.rows[wid].version
        return Snapshot(stock_versions, lane_versions, window_versions)

    def commit(self, plan_obj: Plan, reference: str, snapshot: Snapshot | None = None,
               *, window_ids: dict[int, str] | None = None) -> CommitReceipt:
        if not reference:
            raise ValueError("commit reference required")
        if window_ids is not None:
            plan_obj = _plan_with_window_ids(plan_obj, window_ids)
        moves = list(plan_obj.moves)
        if snapshot is None:
            snapshot = self.snapshot(plan_obj)

        with self.lock:
            if reference in self._committed:
                raise ValueError("duplicate commit reference")

            with self.stock.lock, self.lanes.lock, self.windows.lock:
                stock_out_qty: dict[tuple[str, str], Decimal] = {}
                stock_in_qty: dict[tuple[str, str], Decimal] = {}
                lane_used: dict[tuple[str, str, str | None], Decimal] = {}
                total_window: dict[str, Decimal] = {}

                # ---- Phase 1: aggregate and validate everything ----
                for m in moves:
                    if m.quantity <= 0:
                        raise ValueError(f"move {m.move_id} must be positive")
                    if (m.source, m.sku) not in self.stock.rows:
                        raise KeyError(f"unknown stock {m.source}/{m.sku}")
                    if (m.target, m.sku) not in self.stock.rows:
                        raise KeyError(f"unknown stock {m.target}/{m.sku}")
                    stock_out_qty[(m.source, m.sku)] = (
                        stock_out_qty.get((m.source, m.sku), ZERO) + m.quantity
                    )
                    stock_in_qty[(m.target, m.sku)] = (
                        stock_in_qty.get((m.target, m.sku), ZERO) + m.quantity
                    )
                    lane = resolve_lane(self.lanes, m.source, m.target, m.sku)
                    lane_used[lane.key] = lane_used.get(lane.key, ZERO) + m.quantity
                    wid = window_id(plan_obj, m)
                    win = self.windows.rows.get(wid)
                    if win is None:
                        raise KeyError(f"unknown window {wid}")
                    if win.site != m.target:
                        raise ValueError(
                            f"move {m.move_id} window site {win.site} != target {m.target}"
                        )
                    total_window[wid] = total_window.get(wid, ZERO) + m.quantity

                stock_keys = set(stock_out_qty) | set(stock_in_qty)
                for key in stock_keys:
                    out_qty = stock_out_qty.get(key, ZERO)
                    row = self.stock.rows[key]
                    if key in snapshot.stock and row.version != snapshot.stock[key]:
                        raise RuntimeError(f"stock version conflict at {key}")
                    if row.available < out_qty:
                        raise ValueError(f"insufficient stock at {key}")
                for key, used in lane_used.items():
                    row = self.lanes.rows[key]
                    if key in snapshot.lanes and row.version != snapshot.lanes[key]:
                        raise RuntimeError(f"lane version conflict at {key}")
                    if row.available < used:
                        raise ValueError(f"insufficient lane capacity at {key}")
                for wid, used in total_window.items():
                    row = self.windows.rows[wid]
                    if wid in snapshot.windows and row.version != snapshot.windows[wid]:
                        raise RuntimeError(f"window version conflict at {wid}")
                    if row.available < used:
                        raise ValueError(f"receiving capacity exceeded at {wid}")

                # ---- Phase 2: mutate, only after every check passed ----
                stock_out: list[StockRecord] = []
                for key in sorted(stock_keys):
                    row = self.stock.rows[key]
                    delta = stock_in_qty.get(key, ZERO) - stock_out_qty.get(key, ZERO)
                    nxt = replace(row, available=row.available + delta,
                                  version=row.version + 1)
                    self.stock.rows[key] = nxt
                    stock_out.append(nxt)

                lane_out: list[LaneRecord] = []
                for key, used in lane_used.items():
                    row = self.lanes.rows[key]
                    nxt = replace(row, available=row.available - used,
                                  version=row.version + 1)
                    self.lanes.rows[key] = nxt
                    lane_out.append(nxt)

                window_out: list[WindowRecord] = []
                for wid, used in total_window.items():
                    row = self.windows.rows[wid]
                    nxt = replace(row, available=row.available - used,
                                  version=row.version + 1)
                    self.windows.rows[wid] = nxt
                    window_out.append(nxt)

                self._committed.add(reference)
                return CommitReceipt(
                    reference,
                    tuple(sorted(stock_out, key=lambda r: (r.site, r.sku))),
                    tuple(sorted(lane_out, key=lambda r: r.key)),
                    tuple(sorted(window_out, key=lambda r: r.window_id)),
                    tuple(moves),
                )

    def committed_references(self) -> frozenset[str]:
        with self.lock:
            return frozenset(self._committed)


def _plan_with_window_ids(plan_obj: Plan, window_ids: dict[int, str]) -> Plan:
    return _BoundPlan(plan_obj, window_ids)


class _BoundPlan(Plan):
    """Plan carrying an external window-id mapping; immutable from the
    caller's perspective and behaviourally identical to Plan otherwise."""

    def __init__(self, inner: Plan, window_ids: dict[int, str]):
        object.__setattr__(self, "window_ids", dict(window_ids))
        super().__init__(inner.moves, inner.rejections, inner.residual_gaps)
