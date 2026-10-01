from datetime import timedelta
from .inventory.fefo import reserve
from .quality.release import may_release
from .transport.connections import choose
from .transfer.network_allocation import plan as plan_transfer_network
from .transfer.network_commit import TransferCommitter
class ColdLedgerService:
    def reserve_inventory(self,*args,**kwargs): return reserve(*args,**kwargs)
    def release_decision(self,*args,**kwargs): return may_release(*args,**kwargs)
    def next_connection(self,inbound,candidates,minutes): return choose(inbound,candidates,timedelta(minutes=minutes))
    def plan_transfers(self,*args,**kwargs): return plan_transfer_network(*args,**kwargs)
    def transfer_committer(self,stock,lanes,windows): return TransferCommitter(stock,lanes,windows)
    def health(self): return {'service':'coldledger','status':'ok','runtime_dependencies':0}
