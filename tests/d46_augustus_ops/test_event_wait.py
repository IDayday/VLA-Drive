import threading
import time
from tools.d46_augustus_ops.event_wait import wait_for_events


def test_own_state_publications_do_not_wake_worker(tmp_path):
    def write():
        (tmp_path/'STATE.json').write_text('{}')
        (tmp_path/'STATE.json.tmp').write_text('{}')
    timer=threading.Timer(.03,write);timer.start()
    assert wait_for_events([tmp_path],.12,ignore=('STATE.json',))is False
    timer.join()


def test_native_completion_event_wakes_without_progress_poll(tmp_path):
    timer=threading.Timer(.03,lambda:(tmp_path/'COMPLETE').write_text('{}'));timer.start()
    assert wait_for_events([tmp_path],1.)is True
    timer.join()
