"""Bounded HTTP range access for pinned Parquet dataset files."""
import io
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.session = requests.Session()
        self.session.mount('https://', HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        response = self.session.get(url, headers={'Range': 'bytes=0-0'}, timeout=120, stream=True)
        response.raise_for_status()
        if response.status_code != 206:
            response.close()
            raise ValueError('Server did not honor range request; refusing full download')
        self.url = response.url
        self.size = int(response.headers['Content-Range'].split('/')[-1])
        response.close()
        self.position = 0
        self.cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError('Negative seek')
        return self.position

    def read(self, count=-1):
        end = self.size if count < 0 else min(self.position + count, self.size)
        start = self.position
        if end <= start:
            return b''
        key = (start, end)
        if key not in self.cache:
            response = self.session.get(self.url, headers={'Range': f'bytes={start}-{end-1}'}, timeout=180, stream=True)
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end-1}/{self.size}':
                response.close()
                raise ValueError('Invalid HTTP range response')
            content = response.content
            response.close()
            if len(content) != end-start:
                raise ValueError('Truncated HTTP range response')
            # Cache only small metadata reads, not whole image columns.
            if end-start < 1024*1024:
                self.cache[key] = content
            result = content
        else:
            result = self.cache[key]
        self.position = end
        return result

    def close(self):
        self.session.close()
        super().close()
