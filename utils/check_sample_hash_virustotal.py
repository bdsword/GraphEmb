#!/usr/bin/env python3

import requests
import hashlib
import os
import sys
import re
from datetime import datetime
import time


# Please set your virustotal apikey here.
apikey = None 

src_dir = sys.argv[1]

if apikey is None:
    print('Please set your virustotal apikey first.')
    sys.exit(-1)

if len(sys.argv) != 2:
    print('Usage:\n\t{} <samples_dir>'.format(sys.argv[0]))
    sys.exit(-2)

cur = 0
jump = 64
count = 0
for item in os.listdir(src_dir):
    if cur < jump:
        cur += 1
        continue
    if count == 4:
        time_diff = (datetime.now() - old_time).total_seconds()
        time_to_sleep = 60 - time_diff + 10 # add 10 seconds to prevent error
        time.sleep(time_to_sleep)
        count = 0
    if count == 0:
        old_time = datetime.now()
    with open(os.path.join(src_dir, item), 'rb') as f:
        cont = f.read()
        m = hashlib.sha256()
        m.update(cont)
        digest = m.hexdigest()
        params = {'apikey': apikey, 'resource': digest}
        headers = {
          "Accept-Encoding": "gzip, deflate",
          "User-Agent" : "gzip,  My Python requests library example client or username"
          }
        response = requests.get('https://www.virustotal.com/vtapi/v2/file/report',
                                params=params, headers=headers)
        json_response = response.json()
        print(json_response)
        count += 1
    cur += 1

