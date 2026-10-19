"""Read-only, spawn-safe exact suffix cache shared by native teacher workers."""
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import mmap
from pathlib import Path


def _sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


class ExactSuffixStore:
    @classmethod
    def write(cls,root,values):
        root=Path(root);root.mkdir(parents=True,exist_ok=False)
        offsets={}
        with (root/'values.bin').open('xb') as stream:
            for cursor,value in sorted(values.items()):
                numerator=value.numerator.to_bytes((value.numerator.bit_length()+8)//8,'big',signed=True)
                denominator=value.denominator.to_bytes((value.denominator.bit_length()+7)//8,'big')
                offsets[str(cursor)]=[stream.tell(),len(numerator),len(denominator)]
                stream.write(numerator);stream.write(denominator)
        metadata=dict(protocol='FG_S3_EXACT_SUFFIX_CACHE_V1',values_sha256=_sha(root/'values.bin'),offsets=offsets)
        (root/'index.json').write_text(json.dumps(metadata,sort_keys=True)+'\n')
        return cls(root,_sha(root/'index.json'))

    def __init__(self,root,index_sha256):
        self.root=Path(root);self.index_sha256=index_sha256
        if _sha(self.root/'index.json')!=index_sha256:raise ValueError('Exact suffix cache index changed')
        index=json.loads((self.root/'index.json').read_text())
        if index.get('protocol')!='FG_S3_EXACT_SUFFIX_CACHE_V1' or _sha(self.root/'values.bin')!=index['values_sha256']:
            raise ValueError('Exact suffix cache values changed')
        self.offsets=index['offsets']
        with (self.root/'values.bin').open('rb') as stream:self.mapping=mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)
        self.get=lru_cache(maxsize=128)(self._read)

    def _read(self,cursor):
        offset,nsize,dsize=self.offsets[str(cursor)]
        numerator=int.from_bytes(self.mapping[offset:offset+nsize],'big',signed=True)
        denominator=int.from_bytes(self.mapping[offset+nsize:offset+nsize+dsize],'big')
        # Only this writer's already-reduced Fractions reach this hashed,
        # read-only table. Do not redo huge-integer GCDs on every lookup.
        return Fraction(numerator,denominator,_normalize=False)

    def __getstate__(self):return dict(root=str(self.root),index_sha256=self.index_sha256)
    def __setstate__(self,state):self.__init__(**state)

    def close(self):
        self.get.cache_clear();self.mapping.close()
