import os
import json
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from compare.wsp.harness import Session

S = Session(sys.argv[1])
print(json.dumps(S.final(), indent=1))
