"""Run one unchanged scorer and record waited-child CPU usage with stdlib only."""
import argparse,json,os,resource,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:raise ValueError('An explicit child command is required')
    before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();error=None
    try:
        result=subprocess.run(command,check=False);code=result.returncode
    except OSError as exc:
        error=str(exc);code=127
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    data={'user_seconds':after.ru_utime-before.ru_utime,'system_seconds':after.ru_stime-before.ru_stime,
          'max_rss_kb':after.ru_maxrss,'wall_seconds':time.monotonic()-start,'exit_code':code,
          'mechanism':'Linux resource.RUSAGE_CHILDREN after subprocess wait; includes waited scorer descendants',
          'command':command,'startup_error':error,'scorer_semantics_changed':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);temp=a.output.with_name(a.output.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(a.output)
    if error:print(error,file=sys.stderr)
    raise SystemExit(code if code>=0 else 128-code)


if __name__=='__main__':main()
