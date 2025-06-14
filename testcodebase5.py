from dataclasses import dataclass
import asyncio
import time
import weakref
from typing import List, Dict, Optional
from collections import defaultdict

class MetricCollector:
    def __init__(self):
        self._metrics = {}
        self._callbacks = []

    def register_callback(self, callback):
        print("MetricCollector: Registering callback")
        self._callbacks.append(weakref.ref(callback))

    def record_metric(self, name: str, value: float):
        print(f"MetricCollector: Recording metric {name} with value {value}")
        self._metrics[name] = value
        for cb_ref in self._callbacks:
            callback = cb_ref()
            if callback:
                print(f"MetricCollector: Invoking callback for metric {name}")
                callback(name, value)

class CacheManager:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = defaultdict(int)
        self.max_size = max_size
        
    def get(self, key: str) -> Optional[dict]:
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
        
    def set(self, key: str, value: dict):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        self.cache[key] = value
        self.access_times[key] = time.time()

class DataProcessor:
    def __init__(self):
        self.results = []
        self.processing = False
        self.cache_manager = CacheManager()
        self.metric_collector = MetricCollector()
        
    async def process_batch(self, items: List[Dict]):
        self.processing = True
        tasks = []
        
        for item in items:
            if cached := self.cache_manager.get(str(item['id'])):
                self.results.append(cached)
                continue
                
            task = asyncio.create_task(self._process_item(item))
            tasks.append(task)
            
        await asyncio.gather(*tasks)
        self.processing = False
        return self.results
    
    async def _process_item(self, item: Dict):
        await asyncio.sleep(0.01)  # Simulate processing time
        processed = {
            'id': item['id'],
            'value': item['value'] * 2 if item['value'] > 0 else 0,
            'processed_at': time.time()
        }
        self.cache_manager.set(str(item['id']), processed)
        self.results.append(processed)
        self.metric_collector.record_metric(f"processed_{item['id']}", processed['value'])

class StateManager:
    def __init__(self):
        self._states = {}
        self._transitions = defaultdict(list)
        self._current_state = None
        
    def add_transition(self, from_state: str, to_state: str, condition: callable):
        self._transitions[from_state].append((to_state, condition))
        
    def set_state(self, state: str, data: Dict = None):
        if state not in self._states:
            self._states[state] = data or {}
        
        if self._current_state:
            transitions = self._transitions[self._current_state]
            for next_state, condition in transitions:
                if condition(data):
                    self._current_state = next_state
                    return
                    
        self._current_state = state

class QueueHandler:
    def __init__(self, max_size: int = 100):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.processing = False
        self._consumers = []
        
    async def push(self, item: Dict):
        await self.queue.put(item)
        
    def add_consumer(self, consumer: callable):
        self._consumers.append(consumer)
        
    async def start_processing(self):
        self.processing = True
        while self.processing:
            if self.queue.empty():
                await asyncio.sleep(0.1)
                continue
                
            item = await self.queue.get()
            tasks = [consumer(item) for consumer in self._consumers]
            await asyncio.gather(*tasks)
            self.queue.task_done()

class Application:
    def __init__(self):
        self.processor = DataProcessor()
        self.state_manager = StateManager()
        self.queue_handler = QueueHandler()
        self._setup_state_transitions()
        
    def _setup_state_transitions(self):
        self.state_manager.add_transition(
            'idle', 'processing',
            lambda x: x and x.get('items_count', 0) > 0
        )
        self.state_manager.add_transition(
            'processing', 'idle',
            lambda x: not self.processor.processing
        )
        
    async def process_items(self, items: List[Dict]):
        self.state_manager.set_state('idle', {'items_count': len(items)})
        
        for item in items:
            await self.queue_handler.push(item)
            
        self.queue_handler.add_consumer(self._process_single_item)
        await self.queue_handler.start_processing()
        
        return await self.processor.process_batch(items)
    
    async def _process_single_item(self, item: Dict):
        self.state_manager.set_state('processing', {'current_item': item['id']})
        await asyncio.sleep(0.05)  # Simulate processing time

async def main():
    app = Application()
    test_items = [
        {'id': i, 'value': i * 1.5} 
        for i in range(5)
    ]
    
    results = await app.process_items(test_items)
    print(f"Processed {len(results)} items")

if __name__ == "__main__":
    asyncio.run(main())
