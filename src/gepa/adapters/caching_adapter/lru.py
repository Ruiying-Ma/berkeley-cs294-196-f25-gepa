m_key_timestamp = dict()

def evict(cache_snapshot, obj):
    '''
    This function defines how the algorithm chooses the eviction victim.
    - Args:
        - `cache_snapshot`: A snapshot of the current cache state.
        - `obj`: The new object that needs to be inserted into the cache.
    - Return:
        - `candid_obj_key`: The key of the cached object that will be evicted to make room for `obj`.
    '''
    candid_obj_key = None
    min_ts = min(m_key_timestamp.values())
    candid_obj_keys = list(key for key in cache_snapshot.cache if m_key_timestamp[key] == min_ts)
    candid_obj_key = candid_obj_keys[0]
    return candid_obj_key

def update_after_hit(cache_snapshot, obj):
    '''
    This function defines how the algorithm update the metadata it maintains immediately after a cache hit.
    - Args:
        - `cache_snapshot`: A snapshot of the current cache state.
        - `obj`: The object accessed during the cache hit.
    - Return: `None`
    '''
    global m_key_timestamp
    assert obj.key in m_key_timestamp
    m_key_timestamp[obj.key] = cache_snapshot.access_count

def update_after_insert(cache_snapshot, obj):
    '''
    This function defines how the algorithm updates the metadata it maintains immediately after inserting a new object into the cache.
    - Args:
        - `cache_snapshot`: A snapshot of the current cache state.
        - `obj`: The object that was just inserted into the cache.
    - Return: `None`
    '''
    global m_key_timestamp
    assert obj.key not in m_key_timestamp
    m_key_timestamp[obj.key] = cache_snapshot.access_count

def update_after_evict(cache_snapshot, obj, evicted_obj):
    '''
    This function defines how the algorithm updates the metadata it maintains immediately after evicting the victim.
    - Args:
        - `cache_snapshot`: A snapshot of the current cache state.
        - `obj`: The object to be inserted into the cache.
        - `evicted_obj`: The object that was just evicted from the cache.
    - Return: `None`
    '''
    global m_key_timestamp
    assert obj.key not in m_key_timestamp
    assert evicted_obj.key in m_key_timestamp
    m_key_timestamp.pop(evicted_obj.key)