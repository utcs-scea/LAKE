#https://www.geeksforgeeks.org/lru-cache-in-python-using-ordereddict/
from collections import OrderedDict
 
class LRUCache:
 
    # initialising capacity
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.access = dict()

    # we return the value of the key
    # that is queried in O(1) and return -1 if we
    # don't find the key in out dict / cache.
    # And also move the key to the end
    # to show that it was recently used.
    def get(self, key: int):
        if key not in self.cache:
            return None
        else:
            self.access[key] += 1
            self.cache.move_to_end(key)
            return self.cache[key]
 
    # first, we add / update the key by conventional methods.
    # And also move the key to the end to show that it was recently used.
    # But here we will also check whether the length of our
    # ordered dictionary has exceeded our capacity,
    # If so we remove the first key (least recently used)
    def put(self, key: int, value=0):
        # check if its in our dict
        if key not in self.access:
            self.access[key] = 0

        # return if we had to read from device or not
        io_read = 0
        if key not in self.cache:
            io_read = 1

        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)
        return io_read

    def get_never_access_count(self):
        c = 0
        for k, v in self.access.items():
            if v == 0:
                c += 1
        return c