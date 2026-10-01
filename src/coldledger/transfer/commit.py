from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from threading import RLock
from typing import Iterable

from .model import CommitFailure, TransferPlan


@dataclass(frozen=True)
class CapacityRow:
    key: str
    available: Decimal
    version: int = 0


class CapacityBook:
    """带乐观版本号的容量账本，用于线路与收货窗口两类共享资源。"""

    def __init__(self, rows: Iterable[CapacityRow] = ()):
        self.rows: dict[str, CapacityRow] = {}
        for r in rows:
            self.rows[r.key] = r
        self.lock = RLock()

    def snapshot(self) -> dict[str, CapacityRow]:
        with self.lock:
            return dict(self.rows)

    def ensure(self, key: str) -> CapacityRow:
        with self.lock:
            if key not in self.rows:
                raise KeyError(key)
            return self.rows[key]

    def check(self, key: str, qty: Decimal) -> CapacityRow:
        with self.lock:
            row = self.rows.get(key)
            if row is None:
                raise KeyError(key)
            if qty <= 0:
                raise ValueError("positive quantity required")
            if row.available < qty:
                raise ValueError("insufficient capacity")
            return row

    def take(self, key: str, qty: Decimal) -> CapacityRow:
        with self.lock:
            row = self.rows[key]
            nxt = replace(row, available=row.available - qty, version=row.version + 1)
            self.rows[key] = nxt
            return nxt

    def give_back(self, key: str, qty: Decimal) -> CapacityRow:
        with self.lock:
            row = self.rows[key]
            nxt = replace(row, available=row.available + qty, version=row.version + 1)
            self.rows[key] = nxt
            return nxt


class CommitAborted(RuntimeError):
    """原子提交在校验或任一资源失败时整体中止，所有已做改动已补偿回滚。"""

    def __init__(self, failures: list[CommitFailure]):
        super().__init__("; ".join(f"{f.resource}:{f.key}:{f.reason}" for f in failures))
        self.failures = failures


@dataclass(frozen=True)
class CommitResult:
    plan_id: str
    movement_ids: tuple[str, ...]
    lane_versions: dict[str, int]
    window_versions: dict[str, int]
    ledger_references: tuple[tuple[str, str], ...]


def _window_key(site: str, arrival: date) -> str:
    return f"{site}@{arrival.isoformat()}"


class CommitCoordinator:
    """两阶段提交：
    1) prepare：在同一把全局锁内，对线路、收货窗口、库存台账逐一做版本/容量/余额校验，不产生任何改动；
    2) commit：全部通过后依次落账，任何一步异常都补偿释放此前占用，计划整体不可见、不可半成。
    幂等：同一 plan_id 只能提交一次。
    """

    def __init__(self, lane_book: CapacityBook | None = None,
                 window_book: CapacityBook | None = None,
                 inventory_ledger=None):
        self.lane_book = lane_book
        self.window_book = window_book
        self.ledger = inventory_ledger
        self.lock = RLock()
        self._committed: set[str] = set()

    def commit(self, plan: TransferPlan, actor: str = "planner") -> CommitResult:
        with self.lock:
            if plan.plan_id in self._committed:
                raise RuntimeError(f"plan already committed: {plan.plan_id}")
            if plan.unmet:
                raise CommitAborted([
                    CommitFailure("plan", u.site, f"unmet gap for {u.sku}: {u.remaining}")
                    for u in plan.unmet
                ])
            failures = self._prepare(plan)
            if failures:
                raise CommitAborted(failures)
            return self._commit(plan, actor)

    # ---- phase 1: 全量校验，无副作用 ----
    def _prepare(self, plan: TransferPlan) -> list[CommitFailure]:
        failures: list[CommitFailure] = []
        lane_need: dict[str, Decimal] = {}
        window_need: dict[str, Decimal] = {}
        ledger_need: dict[str, Decimal] = {}

        for m in plan.movements:
            lane_need[m.lane_id] = lane_need.get(m.lane_id, Decimal("0")) + m.quantity
            wk = _window_key(m.target, m.arrival_day)
            window_need[wk] = window_need.get(wk, Decimal("0")) + m.quantity
            ledger_need[m.source] = ledger_need.get(m.source, Decimal("0")) + m.quantity

        if self.lane_book is not None:
            for key, qty in lane_need.items():
                try:
                    self.lane_book.check(key, qty)
                except KeyError:
                    failures.append(CommitFailure("lane", key, "unknown lane"))
                except ValueError as exc:
                    failures.append(CommitFailure("lane", key, str(exc)))

        if self.window_book is not None:
            for key, qty in window_need.items():
                try:
                    self.window_book.check(key, qty)
                except KeyError:
                    failures.append(CommitFailure("window", key, "unknown receiving window"))
                except ValueError as exc:
                    failures.append(CommitFailure("window", key, str(exc)))

        if self.ledger is not None:
            for account, qty in ledger_need.items():
                try:
                    if self.ledger.balance(account) < qty:
                        failures.append(CommitFailure("inventory", account, "insufficient source balance"))
                except Exception as exc:  # 账本不可读同样中止
                    failures.append(CommitFailure("inventory", account, str(exc)))

        return failures

    # ---- phase 2: 全部落账，失败即补偿 ----
    def _commit(self, plan: TransferPlan, actor: str) -> CommitResult:
        taken_lanes: list[tuple[str, Decimal]] = []
        taken_windows: list[tuple[str, Decimal]] = []
        ledger_refs: list[tuple[str, str]] = []
        lane_versions: dict[str, int] = {}
        window_versions: dict[str, int] = {}

        def compensate() -> None:
            for key, qty in reversed(taken_lanes):
                row = self.lane_book.give_back(key, qty)
                lane_versions[key] = row.version
            for key, qty in reversed(taken_windows):
                row = self.window_book.give_back(key, qty)
                window_versions[key] = row.version
            # 库存台账本身支持反向凭证；逐条冲销
            if self.ledger is not None:
                for seq, ref, reason in reversed(getattr(self, "_pending_reversals", [])):
                    try:
                        self.ledger.reverse(seq, f"{plan.plan_id}:rv:{seq}", reason)
                    except Exception:
                        pass

        self._pending_reversals: list[tuple[int, str, str]] = []
        failures: list[CommitFailure] = []

        try:
            lane_totals: dict[str, Decimal] = {}
            for m in plan.movements:
                lane_totals[m.lane_id] = lane_totals.get(m.lane_id, Decimal("0")) + m.quantity
            if self.lane_book is not None:
                for key, qty in lane_totals.items():
                    try:
                        row = self.lane_book.take(key, qty)
                        taken_lanes.append((key, qty))
                        lane_versions[key] = row.version
                    except (KeyError, ValueError) as exc:
                        failures.append(CommitFailure("lane", key, str(exc)))
                        raise CommitAborted(failures) from exc

            window_totals: dict[str, Decimal] = {}
            for m in plan.movements:
                wk = _window_key(m.target, m.arrival_day)
                window_totals[wk] = window_totals.get(wk, Decimal("0")) + m.quantity
            if self.window_book is not None:
                for key, qty in window_totals.items():
                    try:
                        row = self.window_book.take(key, qty)
                        taken_windows.append((key, qty))
                        window_versions[key] = row.version
                    except (KeyError, ValueError) as exc:
                        failures.append(CommitFailure("window", key, str(exc)))
                        raise CommitAborted(failures) from exc

            if self.ledger is not None:
                for m in plan.movements:
                    ref = f"{plan.plan_id}:{m.movement_id}"
                    reason = f"transfer {actor}: {m.source}->{m.target} via {m.lane_id}"
                    try:
                        debit, credit = self.ledger.transfer(m.source, m.target, m.quantity, ref, reason)
                        ledger_refs.append((ref, m.target))
                        self._pending_reversals.append((debit.sequence, ref, reason))
                    except Exception as exc:
                        failures.append(CommitFailure("inventory", m.source, str(exc)))
                        raise CommitAborted(failures) from exc
        except CommitAborted:
            compensate()
            raise

        self._committed.add(plan.plan_id)
        self._pending_reversals = []
        return CommitResult(
            plan_id=plan.plan_id,
            movement_ids=tuple(m.movement_id for m in plan.movements),
            lane_versions=dict(sorted(lane_versions.items())),
            window_versions=dict(sorted(window_versions.items())),
            ledger_references=tuple(ledger_refs),
        )
