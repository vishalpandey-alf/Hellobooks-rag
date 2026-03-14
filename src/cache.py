import hashlib

CACHE = {}

def cache_key(query):
    return hashlib.md5(query.encode()).hexdigest()

def get_cached(query):
    return CACHE.get(cache_key(query))

def set_cache(query,response):
    CACHE[cache_key(query)] = response