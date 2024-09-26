import numpy as np
from multiprocessing import Event, sharedctypes, Value, Lock
import ctypes
import time


class SharedImage:
    def __init__(self, shape, dtype=np.uint8, debug=False):
        self.shape = shape
        self.dtype = dtype
        self.debug = debug
        self.shared_array = sharedctypes.RawArray(ctypes.c_uint8, int(np.prod(self.shape)))

        self.consumed_event = Event()
        self.produced_event = Event()

        self.consumed_event.set()
        self.produced_event.clear()

    def get(self):
        self.produced_event.wait()
        data = np.frombuffer(self.shared_array, dtype=self.dtype).reshape(self.shape)
        self.produced_event.clear()
        self.consumed_event.set()
        return data
    
    def put(self, frame):
        self.consumed_event.wait()
        frame = frame.ravel()
        temp = np.frombuffer(self.shared_array, dtype=self.dtype)
        temp[:] = frame
        self.consumed_event.clear()
        self.produced_event.set()


class SharedImageQueue:
    #TODO: DEBUG
    def __init__(self, shape, dtype=np.uint8, length=20, debug=False):
        self.shape = shape
        self.dtype = dtype
        self.debug = debug
        self.write_count = 0
        self.read_count = 0
        self.length = length

        # 创建共享内存缓冲区
        self.shared_arrays = [sharedctypes.RawArray(ctypes.c_uint8, int(np.prod(self.shape))) for i in range(self.length)]


        self.consumed_events = [Event() for i in range(self.length)]
        self.produced_events = [Event() for i in range(self.length)]
        
        self.read_buffer_idx = Value(ctypes.c_int, 0)
        self.write_buffer_idx = Value(ctypes.c_int, 0)

        for produced_event in self.produced_events:
            produced_event.clear()
        for consumed_event in self.consumed_events:
            consumed_event.set()


    def put(self, frame, debug=False):

        t0 = time.time()
        # if debug:
        #     print(f"{debug} waiting put\n {self.write_buffer_idx.value}, {self.read_buffer_idx.value}\n {[self.produced_events[i].is_set() for i in range(self.length)]} \n {[self.consumed_events[i].is_set() for i in range(self.length)]}")
        
        self.consumed_events[self.write_buffer_idx.value].wait()
        self.produced_events[self.write_buffer_idx.value].clear()

        frame = frame.ravel()
        temp = np.frombuffer(self.shared_arrays[self.write_buffer_idx.value], dtype=self.dtype)
        temp[:] = frame

        self.consumed_events[self.write_buffer_idx.value].clear()
        self.produced_events[self.write_buffer_idx.value].set()
        # if self.write_buffer_idx == 0:
        #     self.event_produced_buffer_0.set()
        #     self.event_consumed_buffer_0.clear()
        # else:
        #     self.event_produced_buffer_1.set()
        #     self.event_consumed_buffer_1.clear()
        if self.debug:
            self.write_count += 1
            # print("write", self.write_count)
        
        # self.write_buffer_idx = (self.write_buffer_idx + 1) % self.length
        self.write_buffer_idx.value = (self.write_buffer_idx.value + 1) % self.length
        # if debug:
        #     print(f"{debug} put\n {self.write_buffer_idx.value}, {self.read_buffer_idx.value}\n {[self.produced_events[i].is_set() for i in range(self.length)]} \n {[self.consumed_events[i].is_set() for i in range(self.length)]}")
        
        # self.write_buffer_idx = 1 - self.write_buffer_idx
        # if debug:
        #     print(f"{debug} put time: {time.time() - t0}")
        #     if debug == "transform_result":
        #         print([self.produced_events[i].is_set() for i in range(self.length)])
        #         print([self.consumed_events[i].is_set() for i in range(self.length)])
            
    def get(self, debug):
        t0 = time.time()
        # if debug:
        #     print(f"{debug} waiting get\n {self.write_buffer_idx.value}, {self.read_buffer_idx.value}\n {[self.produced_events[i].is_set() for i in range(self.length)]} \n {[self.consumed_events[i].is_set() for i in range(self.length)]}")
       
        self.produced_events[self.read_buffer_idx.value].wait()
        # if self.read_buffer_idx == 0:
        #     self.event_produced_buffer_0.wait()
            
        # else:
        #     self.event_produced_buffer_1.wait()

        data = np.frombuffer(self.shared_arrays[self.read_buffer_idx.value], dtype=self.dtype).reshape(self.shape)

        self.produced_events[self.read_buffer_idx.value].clear()
        self.consumed_events[self.read_buffer_idx.value].set()
        # if self.read_buffer_idx == 0:
        #     self.event_produced_buffer_0.clear()
        #     self.event_consumed_buffer_0.set()
        # else:
        #     self.event_produced_buffer_1.clear()
        #     self.event_consumed_buffer_1.set()

        self.read_buffer_idx.value = (self.read_buffer_idx.value + 1) % self.length
        if self.debug:
            self.read_count += 1
            # print("read", self.read_count)
        # if debug:
        #     print(f"{debug} got\n {self.write_buffer_idx.value}, {self.read_buffer_idx.value}\n {[self.produced_events[i].is_set() for i in range(self.length)]} \n {[self.consumed_events[i].is_set() for i in range(self.length)]}")
        # if debug:
        #     print(f"{debug} get time: {time.time() - t0}")
        #     if debug == "push_stream":
        #         print([self.produced_events[i].is_set() for i in range(self.length)])
        #         print([self.consumed_events[i].is_set() for i in range(self.length)])
        return data



if __name__ == '__main__':
    pass
    # size = (2560, 1440)

    # input_urls = (INPUT_STREAM_LEFT_TEST, INPUT_STREAM_RIGHT_TEST)
    # input_urls = (LOCAL_TEST_PATH_LEFT, LOCAL_TEST_PATH_RIGHT)
    # output_url = "test.ts"
    # test_detect_pipeline(SHAPE, input_urls, output_url)
    # test_detect_pipeline(size, input_urls, output_url)