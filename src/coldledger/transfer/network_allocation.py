from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Iterable

ZERO = Decimal("0")
ONE = Decimal("1")

# Structured reason codes recorded for every candidate the global plan
# cannot adopt. Keeping them on the plan (instead of raising) is what makes
# site conflicts auditable after the fact.
REASON_NO_LANE = "no_lane"
REASON_LANE_DISABLED = "lane_disabled"
REASON_SKU_NOT_QUALIFIED = "sku_not_qualified"
REASON_SHELF_LIFE = "shelf_life_below_minimum"
REASON_SOURCE_DRY = "source_has_no_eligible_stock"
REASON_SAFETY_LOCKED = "source_safety_stock_locked"
REASON_LANE_CAPACITY = "lane_capacity_exhausted"
REASON_NO_WINDOW = "no_receiving_window"
REASON_WINDOW_CAPACITY = "receiving_capacity_exhausted"
REASON_SATISFIED = "need_satisfied_elsewhere"
REASON_NO_DEMAND = "unclassified_shortfall"

# Precedence used when several blockers bind the same (need, source).
_STRUCTURAL_ORDER = (
    REASON_LANE_DISABLED,
    REASON_SKU_NOT_QUALIFIED,
    REASON_SAFETY_LOCKED,
    REASON_SOURCE_DRY,
    REASON_SHELF_LIFE,
    REASON_LANE_CAPACITY,
    REASON_NO_WINDOW,
    REASON_WINDOW_CAPACITY,
)


def _d(value) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


@dataclass(frozen=True)
class Lot:
    lot: str
    sku: str
    quantity: Decimal
    expiry: date
    released: bool = True
    quarantined: bool = False

    def validate(self) -> None:
        if not self.lot or not self.sku:
            raise ValueError("lot and sku required")
        if self.quantity < 0:
            raise ValueError("lot quantity must be non-negative")


@dataclass(frozen=True)
class SitePosition:
    site: str
    sku: str
    on_hand: Decimal
    safety_stock: Decimal = ZERO
    forecast_demand: Decimal = ZERO
    velocity_per_day: Decimal = ZERO
    min_inbound_days: int = 0

    def validate(self) -> None:
        if not self.site or not self.sku:
            raise ValueError("site and sku required")
        if self.on_hand < 0 or self.safety_stock < 0 or self.forecast_demand < 0:
            raise ValueError("quantities must be non-negative")
        if self.min_inbound_days < 0:
            raise ValueError("min inbound shelf days must be non-negative")

    @property
    def gap(self) -> Decimal:
        """Uncovered shortfall against safety stock plus forecast demand."""
        return max(ZERO, self.safety_stock + self.forecast_demand - self.on_hand)

    @property
    def shippable(self) -> Decimal:
        """Surplus releasable without pushing the source below safety stock."""
        return max(ZERO, self.on_hand - self.safety_stock)

    @property
    def projected_surplus(self) -> Decimal:
        """What stays releasable after the source covers its own forecast."""
        return max(ZERO, self.on_hand - self.safety_stock - self.forecast_demand)


@dataclass(frozen=True)
class Lane:
    source: str
    target: str
    cost_per_unit: Decimal
    lead_days: int
    capacity: Decimal
    sku: str | None = None
    enabled: bool = True
    reference_cost: Decimal = ONE

    def validate(self) -> None:
        if not self.source or not self.target or self.source == self.target:
            raise ValueError("lane needs two distinct sites")
        if self.cost_per_unit < 0 or self.capacity < 0:
            raise ValueError("cost and capacity must be non-negative")
        if self.lead_days < 0:
            raise ValueError("lead days must be non-negative")
        if self.reference_cost <= 0:
            raise ValueError("reference cost must be positive")

    def carries(self, sku: str) -> bool:
        return self.sku is None or self.sku == sku

    @property
    def label(self) -> str:
        return f"{self.source}->{self.target}"


@dataclass(frozen=True)
class ReceivingWindow:
    site: str
    opens: date
    closes: date
    capacity: Decimal
    sku: str | None = None

    def validate(self) -> None:
        if not self.site:
            raise ValueError("window needs a site")
        if self.closes < self.opens:
            raise ValueError("window closes before it opens")
        if self.capacity < 0:
            raise ValueError("window capacity must be non-negative")

    def accepts(self, sku: str, arrival: date) -> bool:
        return (self.sku is None or self.sku == sku) and self.opens <= arrival <= self.closes


@dataclass(frozen=True)
class Weights:
    """Per-unit reward weights for the five planning factors. The
    highest-scoring candidate wins, so gap severity and expiry exposure
    pull a move in while source safety risk, lane cost/time and tight
    receiving slack push it back."""
    gap: Decimal = Decimal("10")
    expiry: Decimal = Decimal("6")
    source_safety: Decimal = Decimal("4")
    lane_cost: Decimal = Decimal("5")
    lane_time: Decimal = Decimal("2")
    receiving_slack: Decimal = Decimal("3")


@dataclass(frozen=True)
class PlannedMove:
    move_id: str
    source: str
    target: str
    sku: str
    quantity: Decimal
    lot: str
    arrival: date
    window_index: int
    score: float
    factors: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Rejection:
    target: str
    sku: str
    source: str
    reason: str
    detail: str
    shortfall: Decimal = ZERO
    lot: str | None = None
    lane: str | None = None


@dataclass(frozen=True)
class ResidualGap:
    target: str
    sku: str
    quantity: Decimal
    dominant_reason: str
    detail: str


@dataclass(frozen=True)
class Plan:
    moves: tuple[PlannedMove, ...]
    rejections: tuple[Rejection, ...]
    residual_gaps: tuple[ResidualGap, ...]

    def move_quantities(self) -> dict[tuple[str, str, str], Decimal]:
        out: dict[tuple[str, str, str], Decimal] = {}
        for m in self.moves:
            key = (m.source, m.target, m.sku)
            out[key] = out.get(key, ZERO) + m.quantity
        return out

    def inbound(self, target: str, sku: str) -> Decimal:
        return sum(
            (m.quantity for m in self.moves if m.target == target and m.sku == sku),
            ZERO,
        )


def _lot_eligible(lot: Lot, sku: str, arrival: date, min_days: int) -> bool:
    return (
        lot.sku == sku
        and lot.released
        and not lot.quarantined
        and lot.quantity > 0
        and (lot.expiry - arrival).days >= min_days
    )


def _expiry_credit(lot: Lot, arrival: date, velocity_per_day: Decimal) -> float:
    """0..1 expiry exposure. Calibrated so 0 days -> 1, ~3 weeks -> ~0.9,
    ~6 weeks -> ~0.75 and long-dated stock (~8 months) -> ~0.2. Destination
    consumability is enforced separately via min inbound shelf life, while
    fast consumption adds a small extra credit."""
    days = max(0, (lot.expiry - arrival).days)
    urgency = 1.0 / (1.0 + (days / 96.0) ** 1.45)
    if velocity_per_day > 0 and lot.quantity > 0:
        consumable = min(1.0, float(Decimal(days) * velocity_per_day / lot.quantity))
        return min(1.0, urgency + 0.1 * consumable * (1.0 - urgency))
    return urgency


def _source_safety_credit(headroom: Decimal, bite: Decimal) -> float:
    """1 when the source keeps comfortable surplus after this move,
    tending to 0 as the move drains it down to its safety stock."""
    if headroom <= 0:
        return 0.0
    return float(max(ZERO, ONE - bite / (headroom + bite)))


def _lane_credits(lane: Lane) -> tuple[float, float]:
    cost = float(lane.cost_per_unit / lane.reference_cost)
    return 1.0 / (1.0 + cost), 1.0 / (1.0 + lane.lead_days / 7.0)


def _slack_credit(window: ReceivingWindow, remaining: Decimal, bite: Decimal) -> float:
    if window.capacity <= 0:
        return 0.0
    fill_after = (window.capacity - remaining + bite) / window.capacity
    return float(max(ZERO, ONE - fill_after))


def plan(
    positions: Iterable[SitePosition],
    lots_at_site: dict[tuple[str, str], Iterable[Lot]],
    lanes: Iterable[Lane],
    windows: Iterable[ReceivingWindow],
    *,
    weights: Weights | None = None,
    plan_date: date | None = None,
) -> Plan:
    """Build the whole transfer plan before anything is committed.

    Shortfalls are never served one-by-one in gap order: at every step every
    feasible (need, lane, lot, receiving-window) candidate is scored on all
    five factors and the global best is booked first. Receiving capacity,
    lane capacity, source safety stock and lot quantities are all shared
    resources consumed while searching, so a destination that cannot take
    more is discovered during planning rather than halfway through commit.
    ``lots_at_site`` maps (site, sku) to the lots physically held there.
    """
    w = weights or Weights()
    today = plan_date or date.today()

    pos_rows = list(positions)
    lane_rows = list(lanes)
    window_rows = list(windows)
    for p in pos_rows:
        p.validate()
    for lane in lane_rows:
        lane.validate()
    for win in window_rows:
        win.validate()

    pos: dict[tuple[str, str], SitePosition] = {}
    for p in pos_rows:
        if (p.site, p.sku) in pos:
            raise ValueError("duplicate site/sku position")
        pos[(p.site, p.sku)] = p

    site_lots: dict[tuple[str, str], list[Lot]] = {}
    for (site, sku), bucket in lots_at_site.items():
        for lot in bucket:
            lot.validate()
            if lot.sku != sku:
                raise ValueError(f"lot {lot.lot} sku mismatch")
            if (site, sku) not in pos:
                raise ValueError(f"lot {lot.lot} at unknown position {site}/{sku}")
            site_lots.setdefault((site, sku), []).append(lot)

    needs = [p for p in pos_rows if p.gap > 0]

    inbound: dict[tuple[str, str], list[Lane]] = {}
    for lane in lane_rows:
        for need in needs:
            if lane.target == need.site and lane.carries(need.sku):
                inbound.setdefault((need.site, need.sku), []).append(lane)

    # Mutable shared-resource state consumed while searching.
    remaining_gap = {(p.site, p.sku): p.gap for p in needs}
    lane_left = {id(l): l.capacity for l in lane_rows}
    win_left = {i: win.capacity for i, win in enumerate(window_rows)}
    shipped_from: dict[tuple[str, str], Decimal] = {}
    lot_left: dict[str, Decimal] = {}
    for bucket in site_lots.values():
        for lot in bucket:
            if lot.lot in lot_left:
                raise ValueError(f"duplicate lot id {lot.lot}")
            lot_left[lot.lot] = lot.quantity

    def evaluate(need: SitePosition, lane: Lane):
        site, sku = need.site, need.sku
        if not lane.enabled:
            return None, (REASON_LANE_DISABLED, f"lane {lane.label} disabled")
        if not lane.carries(sku):
            return None, (REASON_SKU_NOT_QUALIFIED, f"lane not qualified for {sku}")
        source = pos.get((lane.source, sku))
        if source is None:
            return None, (REASON_SOURCE_DRY, f"source {lane.source} holds no {sku}")
        arrival = today + timedelta(days=lane.lead_days)
        held = site_lots.get((lane.source, sku), [])
        open_lots = [
            l for l in held
            if lot_left[l.lot] > 0 and _lot_eligible(l, sku, arrival, need.min_inbound_days)
        ]
        if not open_lots:
            if not held:
                return None, (REASON_SOURCE_DRY, f"source {lane.source} has no {sku} lots")
            return None, (REASON_SHELF_LIFE, "no lot meets minimum inbound shelf life")
        headroom = source.projected_surplus - shipped_from.get((lane.source, sku), ZERO)
        if headroom <= 0:
            return None, (REASON_SAFETY_LOCKED,
                          "releasable surplus exhausted by safety stock")
        gap = remaining_gap[(site, sku)]
        lot = min(open_lots, key=lambda l: (l.expiry, l.lot))

        fitting_windows = [
            (i, win) for i, win in enumerate(window_rows)
            if win.site == site and win.accepts(sku, arrival)
        ]
        if not fitting_windows:
            return None, (REASON_NO_WINDOW, f"no receiving window on {arrival}")
        open_windows = [(i, win) for i, win in fitting_windows if win_left[i] > 0]
        if not open_windows:
            return None, (REASON_WINDOW_CAPACITY, "receiving capacity exhausted")

        # Largest remaining room defines the bite; ties go to the window
        # closing first so slack stays available for later arrivals.
        i, win = max(open_windows, key=lambda x: (win_left[x[0]], -x[1].closes.toordinal(), -x[0]))

        # The bite is bounded by every shared resource at once, including
        # the destination's receiving room — this is what prevents planning
        # more than a site can physically receive.
        bite = min(gap, headroom, lane_left[id(lane)], lot_left[lot.lot], win_left[i])
        if bite <= 0:
            if lane_left[id(lane)] <= 0:
                return None, (REASON_LANE_CAPACITY, "lane capacity exhausted")
            return None, (REASON_SOURCE_DRY, "no releasable quantity")

        gap_credit = float(gap / (need.safety_stock + need.forecast_demand))
        expiry_credit = _expiry_credit(lot, arrival, need.velocity_per_day)
        safety_credit = _source_safety_credit(headroom, bite)
        cost_credit, time_credit = _lane_credits(lane)
        slack = _slack_credit(win, win_left[i], bite)
        factors = {
            "gap": gap_credit,
            "expiry": expiry_credit,
            "source_safety": safety_credit,
            "lane_cost": cost_credit,
            "lane_time": time_credit,
            "receiving_slack": slack,
        }
        score = (
            float(w.gap) * gap_credit
            + float(w.expiry) * expiry_credit
            + float(w.source_safety) * safety_credit
            + float(w.lane_cost) * cost_credit
            + float(w.lane_time) * time_credit
            + float(w.receiving_slack) * slack
        )
        return {
            "need": need,
            "lane": lane,
            "lot": lot,
            "arrival": arrival,
            "window": i,
            "bite": bite,
            "score": score,
            "factors": factors,
        }, None

    moves: list[PlannedMove] = []
    move_seq = 0
    max_iterations = len(lot_left) + len(needs) * len(lane_rows) + 1

    for _ in range(max_iterations):
        best = None
        for need in needs:
            if remaining_gap[(need.site, need.sku)] <= 0:
                continue
            for lane in inbound.get((need.site, need.sku), ()):
                candidate, _ = evaluate(need, lane)
                if candidate is None:
                    continue
                # Deterministic total order: score first, then stable keys.
                key = (
                    -candidate["score"],
                    need.site,
                    need.sku,
                    lane.source,
                    candidate["lot"].lot,
                )
                if best is None or key < best[0]:
                    best = (key, candidate)
        if best is None:
            break

        _, c = best
        move_seq += 1
        moves.append(
            PlannedMove(
                move_id=f"M{move_seq:04d}",
                source=c["lane"].source,
                target=c["need"].site,
                sku=c["need"].sku,
                quantity=c["bite"],
                lot=c["lot"].lot,
                arrival=c["arrival"],
                window_index=c["window"],
                score=c["score"],
                factors=c["factors"],
            )
        )
        remaining_gap[(c["need"].site, c["need"].sku)] -= c["bite"]
        lane_left[id(c["lane"])] -= c["bite"]
        win_left[c["window"]] -= c["bite"]
        src_key = (c["lane"].source, c["need"].sku)
        shipped_from[src_key] = shipped_from.get(src_key, ZERO) + c["bite"]
        lot_left[c["lot"].lot] -= c["bite"]

    rejections, residuals = _sweep(
        needs,
        pos,
        site_lots,
        lane_rows,
        window_rows,
        win_left,
        lane_left,
        shipped_from,
        lot_left,
        remaining_gap,
        today,
    )
    return Plan(tuple(moves), tuple(rejections), tuple(residuals))


def _dominant(reasons: list[tuple[str, str]]) -> tuple[str, str]:
    for code in _STRUCTURAL_ORDER:
        for reason in reasons:
            if reason[0] == code:
                return reason
    return reasons[0]


def _sweep(
    needs,
    pos,
    site_lots,
    lane_rows,
    window_rows,
    win_left,
    lane_left,
    shipped_from,
    lot_left,
    remaining_gap,
    today,
):
    """Final audit pass: record why every unadopted (need, source, lot)
    candidate could not be taken, and classify every residual gap."""
    rejections: list[Rejection] = []
    seen: set[tuple[str, str, str, str, str | None]] = set()
    residuals: list[ResidualGap] = []

    def record(site, sku, source, code, detail, shortfall=ZERO, lot=None, lane=None):
        key = (site, sku, source, code, lot)
        if key in seen:
            return
        seen.add(key)
        lane_name = lane.label if isinstance(lane, Lane) else lane
        rejections.append(
            Rejection(site, sku, source, code, detail, _d(shortfall), lot, lane_name)
        )

    for need in needs:
        site, sku = need.site, need.sku
        gap = remaining_gap[(site, sku)]
        lanes_here = [l for l in lane_rows if l.target == site]
        sources = {l.source for l in lanes_here}
        other_sites = sorted(
            s for (s, k) in pos if k == sku and s != site and s not in sources
        )

        if not lanes_here:
            record(site, sku, "*", REASON_NO_LANE, "no inbound lane configured", gap)
        for source in other_sites:
            record(site, sku, source, REASON_NO_LANE, "no lane from source site", gap)

        dominant_for_need: list[tuple[str, str]] = []
        for source in sorted(sources):
            per_source: list[tuple[str, str]] = []
            for lane in (l for l in lanes_here if l.source == source):
                arrival = today + timedelta(days=lane.lead_days)
                source_pos = pos.get((source, sku))
                if not lane.enabled:
                    reason = (REASON_LANE_DISABLED, f"lane {lane.label} disabled")
                    record(site, sku, source, *reason, gap, lane=lane)
                    per_source.append(reason)
                    continue
                if not lane.carries(sku):
                    reason = (REASON_SKU_NOT_QUALIFIED, f"lane not qualified for {sku}")
                    record(site, sku, source, *reason, gap, lane=lane)
                    per_source.append(reason)
                    continue
                if source_pos is None:
                    reason = (REASON_SOURCE_DRY, f"source {source} holds no {sku}")
                    record(site, sku, source, *reason, gap, lane=lane)
                    per_source.append(reason)
                    continue

                headroom = source_pos.projected_surplus - shipped_from.get((source, sku), ZERO)
                held = site_lots.get((source, sku), [])
                fitting_windows = [
                    (i, win) for i, win in enumerate(window_rows)
                    if win.site == site and win.accepts(sku, arrival)
                ]

                # Every lot that fails the inbound shelf-life rule keeps its
                # own rejection row, even when a sibling lot is adopted.
                ineligible = [
                    l for l in held
                    if not l.released
                    or l.quarantined
                    or (l.expiry - arrival).days < need.min_inbound_days
                ]
                eligible = [
                    l for l in held if _lot_eligible(l, sku, arrival, need.min_inbound_days)
                ]
                for l in sorted(ineligible, key=lambda x: (x.expiry, x.lot)):
                    if not l.released or l.quarantined:
                        detail = f"lot {l.lot} not releasable (quarantined or held)"
                    else:
                        days = (l.expiry - arrival).days
                        detail = (
                            f"lot {l.lot} expires {l.expiry}; "
                            f"{days}d shelf life < {need.min_inbound_days}d required"
                        )
                    record(site, sku, source, REASON_SHELF_LIFE, detail, gap,
                           lot=l.lot, lane=lane)

                if gap <= 0:
                    reason = (REASON_SATISFIED, "need fully covered by adopted moves")
                elif headroom <= 0:
                    reason = (REASON_SAFETY_LOCKED,
                              "releasable surplus exhausted by safety stock")
                elif not eligible:
                    if not held:
                        reason = (REASON_SOURCE_DRY, f"source {source} has no {sku} lots")
                    else:
                        reason = (REASON_SHELF_LIFE,
                                  "no lot meets minimum inbound shelf life")
                elif lane_left[id(lane)] <= 0:
                    reason = (REASON_LANE_CAPACITY, "lane capacity exhausted by plan")
                elif not fitting_windows:
                    reason = (REASON_NO_WINDOW, f"no receiving window on {arrival}")
                elif all(win_left[i] <= 0 for i, _ in fitting_windows):
                    reason = (REASON_WINDOW_CAPACITY,
                              "receiving capacity exhausted by plan")
                else:
                    # Residual on an otherwise feasible lane: report every
                    # binding limit and let precedence pick the dominant one.
                    binding: list[tuple[str, str]] = []
                    if headroom < gap:
                        binding.append((REASON_SAFETY_LOCKED,
                                        f"only {headroom} releasable before safety stock"))
                    if lane_left[id(lane)] < gap:
                        binding.append((REASON_LANE_CAPACITY,
                                        f"only {lane_left[id(lane)]} lane capacity left"))
                    win_cap = sum(
                        (win_left[i] for i, _ in fitting_windows if win_left[i] > 0),
                        ZERO,
                    )
                    if win_cap < gap:
                        binding.append((REASON_WINDOW_CAPACITY,
                                        f"only {win_cap} receiving capacity left"))
                    lot_qty = sum(
                        (lot_left.get(l.lot, ZERO) for l in held
                         if lot_left.get(l.lot, ZERO) > 0
                         and _lot_eligible(l, sku, arrival, need.min_inbound_days)),
                        ZERO,
                    )
                    if lot_qty < gap:
                        binding.append((REASON_SOURCE_DRY,
                                        f"only {lot_qty} eligible lot quantity left"))
                    reason = _dominant(binding) if binding else (
                        REASON_NO_DEMAND, "unclassified shortfall")

                record(site, sku, source, *reason, gap, lane=lane)
                per_source.append(reason)

            if per_source:
                dominant_for_need.append(_dominant(per_source))

        if gap > 0:
            code, detail = (
                _dominant(dominant_for_need)
                if dominant_for_need
                else (REASON_NO_LANE, "no inbound lane configured")
            )
            residuals.append(ResidualGap(site, sku, gap, code, detail))

    rejections.sort(key=lambda r: (r.target, r.sku, r.source, r.reason, r.lot or ""))
    residuals.sort(key=lambda g: (g.target, g.sku))
    return rejections, residuals


def explain(plan_obj: Plan) -> list[str]:
    lines = [
        f"MOVE {m.move_id}: {m.source}->{m.target} {m.sku} x{m.quantity} "
        f"lot={m.lot} arrive={m.arrival} score={m.score:.3f}"
        for m in plan_obj.moves
    ]
    lines += [
        f"REJECT {r.source}->{r.target} {r.sku}: {r.reason} ({r.detail})"
        + (f" shortfall={r.shortfall}" if r.shortfall > 0 else "")
        + (f" lot={r.lot}" if r.lot else "")
        for r in plan_obj.rejections
    ]
    lines += [
        f"GAP {g.target} {g.sku} x{g.quantity}: {g.dominant_reason} ({g.detail})"
        for g in plan_obj.residual_gaps
    ]
    return lines
