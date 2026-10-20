"""Process-local batch undo; persistent Reader state remains a separate API."""
from contextlib import contextmanager
import copy
from dataclasses import dataclass, field


@contextmanager
def batch_transaction(owner):
    mark=owner.begin_batch()
    try:
        yield mark
    except BaseException:
        owner.rollback_batch(mark)
        raise
    else:
        owner.commit_batch(mark)


@dataclass(eq=False)
class ReaderMark:
    rng: dict
    counts: object
    market: set = field(default_factory=set)
    account: set = field(default_factory=set)


class ReaderTransactions:
    def __init__(self,reader):
        self.reader=reader;self.stack=[]

    def begin(self):
        mark=ReaderMark(copy.deepcopy(self.reader.rng.bit_generator.state),self.reader.counts.copy())
        self.stack.append(mark)
        return mark

    def _check(self,mark):
        if not self.stack or self.stack[-1] is not mark:
            raise ValueError('READER_BATCH_TRANSACTION_OWNER_OR_ORDER')

    def add(self,market,account):
        if not self.stack:raise ValueError('READER_BATCH_TRANSACTION_REQUIRED')
        r=self.reader;mark=self.stack[-1]
        new_market=market-r.seen_market;new_account=account-r.seen_account
        mark.market.update(new_market);mark.account.update(new_account)
        r.seen_market.update(new_market);r.seen_account.update(new_account)

    def commit(self,mark):
        self._check(mark);self.stack.pop()
        if self.stack:
            self.stack[-1].market.update(mark.market);self.stack[-1].account.update(mark.account)

    def rollback(self,mark):
        self._check(mark)
        r=self.reader
        r.seen_market.difference_update(mark.market);r.seen_account.difference_update(mark.account)
        r.counts=mark.counts.copy();r.rng.bit_generator.state=copy.deepcopy(mark.rng)
        self.stack.pop()
        try:r._guard()
        except BaseException:r._failed=True


class TransactionGroup:
    """Compose child transactions without committing through an outer owner."""
    def __init__(self,owner,children,save_metadata,load_metadata):
        self.owner=owner;self.children=children
        self.save_metadata=save_metadata;self.load_metadata=load_metadata;self.stack=[]

    def begin(self):
        state=copy.deepcopy(self.save_metadata());marks=[]
        try:
            for child in self.children:marks.append((child,child.begin_batch()))
        except BaseException:
            for child,mark in reversed(marks):child.rollback_batch(mark)
            raise
        token=(state,marks);self.stack.append(token)
        return token

    def _check(self,token):
        if not self.stack or self.stack[-1] is not token:
            raise ValueError('STREAM_BATCH_TRANSACTION_OWNER_OR_ORDER')

    def commit(self,token):
        self._check(token)
        for child,mark in reversed(token[1]):child.commit_batch(mark)
        self.stack.pop()

    def rollback(self,token):
        self._check(token);failure=None
        for child,mark in reversed(token[1]):
            try:child.rollback_batch(mark)
            except BaseException as error:failure=error
        self.load_metadata(copy.deepcopy(token[0]));self.stack.pop()
        # A source Guard failure is permanent for the group even though its
        # local counters/RNG can still be undone without rereading that source.
        for child,_ in token[1]:
            readers=getattr(child,'readers',())
            if isinstance(readers,dict):readers=readers.values()
            if (getattr(child,'_failed',False) or getattr(getattr(child,'stream',None),'_failed',False)
                    or any(getattr(r,'_failed',False) for r in readers)):
                self.owner._failed=True
        if failure is not None:
            self.owner._failed=True
            raise RuntimeError('STREAM_TRANSACTION_ROLLBACK_FAILED_REOPEN_REQUIRED') from failure
