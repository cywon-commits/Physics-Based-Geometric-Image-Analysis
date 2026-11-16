import os,time,json
from pathlib import Path
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  try:
   args=(p/'cmdline').read_bytes()
   if b'python\x00code/study.py\x00--train' in args:pid=p;break
  except OSError:pass
else:raise RuntimeError('training process not found')
last=None
while True:
 try:
  s=(pid/'stat').read_text().split();last=dict(pid=int(pid.name),process_CPU_seconds=(int(s[13])+int(s[14]))/os.sysconf('SC_CLK_TCK'),sampling_interval_seconds=.1,scope='whole training process user+system CPU; last sample before exit, undercount at most one sample interval plus scheduling',sample_utc=time.time());Path('output/training_process_cpu.json').write_text(json.dumps(last,indent=2))
  time.sleep(.1)
 except OSError:break
