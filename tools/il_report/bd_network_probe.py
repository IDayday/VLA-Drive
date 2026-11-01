"""Finite CPU-only container TCP throughput measurement; no training changes."""
import argparse
import json
import os
from pathlib import Path
import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor


HEADER=struct.Struct('!Q')


def recv_exact(connection,size):
    parts=[]
    while size:
        block=connection.recv(size)
        if not block:raise ConnectionError('Truncated probe header')
        parts.append(block);size-=len(block)
    return b''.join(parts)


def atomic(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2));os.replace(temporary,path)


def parallel(a):
    cases=[(streams,repeat)for streams in(1,4,8)for repeat in range(2)]
    size=128*1024**2;payload=b'x'*(4*1024**2)
    def receive(connection):
        with connection:
            connection.settimeout(30);n=HEADER.unpack(recv_exact(connection,8))[0]
            if n!=size:raise ValueError('Unexpected parallel payload')
            while n:
                block=connection.recv(min(n,len(payload)))
                if not block:raise ConnectionError('Incomplete parallel payload')
                n-=len(block)
            connection.sendall(b'1')
    if a.server:
        with socket.socket()as listener:
            listener.bind((a.address,0));listener.listen(16);listener.settimeout(60)
            atomic(a.output.with_suffix('.ready.json'),dict(address=a.address,port=listener.getsockname()[1],pid=os.getpid()))
            for streams,_ in cases:
                with ThreadPoolExecutor(max_workers=streams)as pool:
                    futures=[pool.submit(receive,listener.accept()[0])for _ in range(streams)]
                    for f in futures:f.result()
            atomic(a.output,dict(status='COMPLETE',cases=len(cases),payload_bytes=sum(n*size for n,_ in cases),GPU_calls=0))
        return
    def send(_):
        with socket.create_connection((a.address,a.port),timeout=30)as connection:
            connection.settimeout(30);connection.sendall(HEADER.pack(size));remaining=size
            while remaining:
                chunk=min(remaining,len(payload));connection.sendall(memoryview(payload)[:chunk]);remaining-=chunk
            if recv_exact(connection,1)!=b'1':raise ValueError('Missing parallel ACK')
    rows=[]
    for streams,repeat in cases:
        start=time.monotonic()
        with ThreadPoolExecutor(max_workers=streams)as pool:list(pool.map(send,range(streams)))
        seconds=time.monotonic()-start;rows.append(dict(streams=streams,repeat=repeat,bytes=streams*size,seconds=seconds,GiB_s=streams*size/2**30/seconds))
    atomic(a.output,dict(status='COMPLETE',host=socket.gethostname(),destination=a.address,rows=rows,GPU_calls=0,
        scope='Finite aggregate container TCP throughput, not NCCL'))
    print(json.dumps(rows))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--server',action='store_true')
    parser.add_argument('--address',required=True);parser.add_argument('--port',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--parallel',action='store_true');a=parser.parse_args()
    if a.parallel:return parallel(a)
    sizes=[8*1024**2]+[64*1024**2]*3+[512*1024**2]*3
    if a.server:
        with socket.socket()as listener:
            listener.bind((a.address,0));listener.listen(1);listener.settimeout(60)
            atomic(a.output.with_suffix('.ready.json'),dict(address=a.address,port=listener.getsockname()[1],
                pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21])))
            for expected in sizes:
                connection,_=listener.accept()
                with connection:
                    connection.settimeout(30);size=HEADER.unpack(recv_exact(connection,8))[0]
                    if size!=expected:raise ValueError('Unregistered probe size')
                    remaining=size
                    while remaining:
                        block=connection.recv(min(4*1024**2,remaining))
                        if not block:raise ConnectionError('Incomplete finite payload')
                        remaining-=len(block)
                    connection.sendall(b'1')
            atomic(a.output,dict(status='COMPLETE',cases=len(sizes),payload_bytes=sum(sizes),GPU_calls=0))
        return
    payload=b'x'*(4*1024**2);rows=[]
    for index,size in enumerate(sizes):
        with socket.create_connection((a.address,a.port),timeout=30)as connection:
            connection.settimeout(30);connection.sendall(HEADER.pack(size));start=time.monotonic()
            remaining=size
            while remaining:
                chunk=min(remaining,len(payload));connection.sendall(memoryview(payload)[:chunk]);remaining-=chunk
            if recv_exact(connection,1)!=b'1':raise ValueError('Missing payload acknowledgement')
            seconds=time.monotonic()-start
            rows.append(dict(bytes=size,seconds=seconds,GiB_s=size/2**30/seconds,Gbit_s=size*8/10**9/seconds,warmup=index==0))
    atomic(a.output,dict(status='COMPLETE',host=socket.gethostname(),destination=a.address,rows=rows,
        total_payload_bytes=sum(sizes),GPU_calls=0,scope='Actual container TCP payload throughput; not NCCL or RDMA performance'))
    print(json.dumps(rows))


if __name__=='__main__':main()
