import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('LANG', 'ko_KR.UTF-8')
os.environ.setdefault('LC_ALL', 'ko_KR.UTF-8')

bind = "127.0.0.1:5000"
workers = 4
worker_class = "sync"
timeout = 300