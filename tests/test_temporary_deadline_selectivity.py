import json
import types
import unittest
from unittest.mock import patch,Mock
from tools.il_report.temporary_resource_deadline import REMOTE


class DeadlineSelectivity(unittest.TestCase):
    def execute(self,mode,changed=False):
        touched=[];stat_reads={};root='/mnt/project/own'
        command={10:['python','-m','known.train'],11:['python','-m','other.train'],12:['python','-m','known.train']}
        env={10:root+'/jobs/a/STOP_REQUESTED',11:root+'/jobs/b/STOP_REQUESTED',12:'/mnt/project/other/jobs/c/STOP_REQUESTED'}
        class FakePath:
            def __init__(self,value):self.value=str(value)
            def __truediv__(self,x):return FakePath(self.value+'/'+str(x))
            def __str__(self):return self.value
            @property
            def name(self):return self.value.rsplit('/',1)[-1]
            def iterdir(self):return [FakePath('/proc/'+str(p))for p in command]
            def read_text(self):
                pid=int(self.value.split('/')[2]);stat_reads[pid]=stat_reads.get(pid,0)+1
                fields=['0']*22;fields[2]='S';fields[21]=str(999 if changed and stat_reads[pid]>1 else 123)
                return ' '.join(fields)
            def read_bytes(self):
                pid=int(self.value.split('/')[2])
                if self.name=='cmdline':return chr(0).join(command[pid]).encode()
                return ('WORLD_POSTCOT_STOP_FILE='+env[pid]).encode()
            def touch(self,**_):touched.append(self.value)
        request=dict(mode=mode,owned_roots=[root],owned_modules=['known.train'])
        kill=Mock();opened=Mock(return_value=999);printed=[]
        with patch('pathlib.Path',FakePath),patch('os.pidfd_open',opened),patch('os.close'),patch('signal.pidfd_send_signal',kill),patch('builtins.print',side_effect=printed.append):
            exec(REMOTE.replace('__REQUEST__',repr(request)),{})
        return json.loads(printed[-1]),touched,opened,kill

    def test_stop_only_verified_owned_scope(self):
        result,touched,opened,kill=self.execute('stop')
        self.assertEqual([r['pid']for r in result['owned_processes']],[10]);self.assertEqual(touched,['/mnt/project/own/jobs/a/STOP_REQUESTED']);kill.assert_not_called();opened.assert_not_called()

    def test_audit_has_no_mutations(self):
        result,touched,opened,kill=self.execute('audit')
        self.assertEqual(len(result['owned_processes']),1);self.assertEqual(touched,[]);kill.assert_not_called();opened.assert_not_called()

    def test_hard_exit_uses_pidfd_and_rejects_reuse(self):
        result,touched,opened,kill=self.execute('kill');opened.assert_called_once_with(10);self.assertEqual(kill.call_count,1)
        with self.assertRaisesRegex(RuntimeError,'Process changed'):self.execute('kill',changed=True)


if __name__=='__main__':unittest.main()
