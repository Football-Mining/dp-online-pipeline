import time
from functools import wraps


def sub_task(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Starting task:", func.__name__)
        result = func(*args, **kwargs)
        print("Finished task:", func.__name__)
        return result

    return wrapper

def iteration_timer(function_name=None, action_name=None):
    def decorator(func):
        count = 0  # 初始化计数器

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal count  # 使用外部变量
            count += 1
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if function_name is None:
                function_name == ""
            if action_name is None:
                action_name == ""
            # print(f"{function_name} {action_name} iteration {count} took {elapsed_time:.6f} seconds.")
            return result
        return wrapper
    return decorator
