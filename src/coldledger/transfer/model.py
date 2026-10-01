from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum


class RejectReason(str, Enum):
    LOT_INELIGIBLE = "lot_ineligible"                 # 批次未放行或被隔离
    NO_DEMAND = "no_demand"                           # 目的站该 SKU 无缺口
    SOURCE_SAFETY_STOCK = "source_safety_stock"       # 源站可用盈余被安全库存锁死
    SHELF_LIFE = "shelf_life"                         # 到货后效期不足最小可售天数
    NO_LANE = "no_lane"                               # 源站到目的站无可用线路
    NO_WINDOW = "no_receiving_window"                 # 到货日无收货窗口
    LANE_CAPACITY = "lane_capacity"                   # 线路容量被占满
    RECEIVING_CAPACITY = "receiving_capacity"         # 目的站收货窗口容量被占满
    ALTERNATIVE_SELECTED = "alternative_selected"    # 全局方案选择了成本更低的组合
    SHORTAGE_UNFILLED = "shortage_unfilled"           # 缺口最终未被任何方案覆盖


@dataclass(frozen=True)
class StockLot:
    site: str
    sku: str
    lot: str
    expiry: date
    quantity: Decimal
    released: bool = True
    quarantined: bool = False

    def validate(self) -> None:
        if not self.site or not self.sku or not self.lot:
            raise ValueError("site, sku and lot required")
        if self.quantity <= 0:
            raise ValueError("lot quantity must be positive")


@dataclass(frozen=True)
class SiteNeed:
    site: str
    sku: str
    gap: Decimal
    priority: int = 0

    def validate(self) -> None:
        if not self.site or not self.sku:
            raise ValueError("site and sku required")
        if self.gap <= 0:
            raise ValueError("gap must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")


@dataclass(frozen=True)
class Lane:
    lane_id: str
    source: str
    target: str
    cost_per_unit: Decimal
    lead_days: int
    capacity: Decimal
    risk: float = 0.0
    enabled: bool = True

    def validate(self) -> None:
        if not self.lane_id or not self.source or not self.target:
            raise ValueError("lane id and endpoints required")
        if self.source == self.target:
            raise ValueError("self lane")
        if self.cost_per_unit < 0 or self.capacity < 0:
            raise ValueError("negative lane cost or capacity")
        if self.lead_days < 0:
            raise ValueError("negative lead time")
        if not 0.0 <= self.risk <= 1.0:
            raise ValueError("risk outside 0..1")


@dataclass(frozen=True)
class ReceivingWindow:
    site: str
    day: date
    capacity: Decimal

    def validate(self) -> None:
        if not self.site:
            raise ValueError("site required")
        if self.capacity < 0:
            raise ValueError("negative window capacity")


@dataclass(frozen=True)
class Weights:
    """多因子目标函数权重。缺口未满足的代价必须显著高于任何可行调拨路径。"""
    lane: Decimal = Decimal("1")
    lane_risk: Decimal = Decimal("60")
    expiry: Decimal = Decimal("150")
    shortage: Decimal = Decimal("100000")
    priority_step: Decimal = Decimal("0.15")
    expiry_horizon_days: int = 180

    def validate(self) -> None:
        if any(x < 0 for x in (self.lane, self.lane_risk, self.expiry, self.shortage, self.priority_step)):
            raise ValueError("weights must be non-negative")
        if self.expiry_horizon_days <= 0:
            raise ValueError("expiry horizon must be positive")


@dataclass(frozen=True)
class LotPick:
    lot: str
    expiry: date
    quantity: Decimal


@dataclass(frozen=True)
class Movement:
    movement_id: str
    sku: str
    source: str
    target: str
    lane_id: str
    quantity: Decimal
    picks: tuple[LotPick, ...]
    arrival_day: date


@dataclass(frozen=True)
class UnmetGap:
    site: str
    sku: str
    gap: Decimal
    remaining: Decimal
    reason: RejectReason
    detail: str = ""


@dataclass(frozen=True)
class Rejection:
    """每一个被评估但未采用的候选调拨及其原因。"""
    sku: str
    source: str
    target: str
    lot: str
    lane_id: str
    reason: RejectReason
    detail: str = ""


@dataclass(frozen=True)
class TransferPlan:
    plan_id: str
    ship_day: date
    min_remaining_days: int
    movements: tuple[Movement, ...]
    unmet: tuple[UnmetGap, ...]
    rejections: tuple[Rejection, ...]
    total_score: Decimal
    generated_at: datetime
    weights: Weights

    def movement(self, movement_id: str) -> Movement | None:
        for m in self.movements:
            if m.movement_id == movement_id:
                return m
        return None

    def is_complete(self) -> bool:
        return not self.unmet


@dataclass(frozen=True)
class CommitFailure:
    resource: str
    key: str
    reason: str
