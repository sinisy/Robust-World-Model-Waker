import re
import sys


class Flags:

  def __init__(self, *args, **kwargs):
    from .config import Config
    self._config = Config(*args, **kwargs)

  def parse(self, argv=None, known_only=False, help_exists=None):
    if help_exists is None:
      help_exists = not known_only
    if argv is None:
      argv = sys.argv[1:]
    if '--help' in argv:
      print('\nHelp:')
      lines = str(self._config).split('\n')[2:]
      print('\n'.join('--' + re.sub(r'[:,\[\]]', '', x) for x in lines))
      help_exists and sys.exit()
    parsed = {}
    remaining = []
    key = None
    vals = None
    for arg in argv:
      if arg.startswith('--'):
        if key:
          self._submit_entry(key, vals, parsed, remaining)
        if '=' in arg:
          key, val = arg.split('=', 1)
          vals = [val]
        else:
          key, vals = arg, []
      else:
        if key:
          vals.append(arg)
        else:
          remaining.append(arg)
    self._submit_entry(key, vals, parsed, remaining)
    parsed = self._config.update(parsed)
    if known_only:
      return parsed, remaining
    else:
      for flag in remaining:
        if flag.startswith('--'):
          raise ValueError(f"Flag '{flag}' did not match any config keys.")
      assert not remaining, remaining
   